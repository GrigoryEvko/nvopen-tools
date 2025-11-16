// Function: sub_ED7AF0
// Address: 0xed7af0
//
__int64 __fastcall sub_ED7AF0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  int v6; // r11d
  unsigned int v7; // r10d
  __int64 v9; // r14
  unsigned __int64 v10; // rbx
  unsigned int v11; // r12d
  __int64 i; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 result; // rax
  bool v17; // cc
  _QWORD *v18; // rcx
  unsigned int v19; // r15d
  __int64 v20; // rsi
  bool v21; // cc
  _QWORD *v22; // rdx
  __int64 v23; // rcx
  _QWORD *v24; // rcx
  __int64 v25; // [rsp+0h] [rbp-30h]

  v6 = a4;
  v7 = a4;
  v9 = (a3 - 1) / 2;
  v10 = HIDWORD(a4);
  v11 = HIDWORD(a4);
  v25 = a3 & 1;
  if ( a2 >= v9 )
  {
    result = a1 + 16 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_18;
    v13 = a2;
    goto LABEL_24;
  }
  for ( i = a2; ; i = v13 )
  {
    v13 = 2 * (i + 1);
    v14 = 32 * (i + 1);
    v15 = a1 + v14 - 16;
    result = a1 + v14;
    v17 = *(_DWORD *)result <= *(_DWORD *)v15;
    if ( *(_DWORD *)result < *(_DWORD *)v15
      || *(_DWORD *)result == *(_DWORD *)v15
      && (v19 = *(_DWORD *)(v15 + 4), v17 = *(_DWORD *)(result + 4) <= v19, *(_DWORD *)(result + 4) < v19)
      || v17 && *(_QWORD *)(result + 8) < *(_QWORD *)(v15 + 8) )
    {
      --v13;
      result = a1 + 16 * v13;
    }
    v18 = (_QWORD *)(a1 + 16 * i);
    *v18 = *(_QWORD *)result;
    v18[1] = *(_QWORD *)(result + 8);
    if ( v13 >= v9 )
      break;
  }
  if ( !v25 )
  {
LABEL_24:
    if ( (a3 - 2) / 2 == v13 )
    {
      v23 = v13 + 1;
      v13 = 2 * (v13 + 1) - 1;
      v24 = (_QWORD *)(a1 + 32 * v23 - 16);
      *(_QWORD *)result = *v24;
      *(_QWORD *)(result + 8) = v24[1];
      result = a1 + 16 * v13;
    }
  }
  v20 = (v13 - 1) / 2;
  if ( v13 > a2 )
  {
    while ( 1 )
    {
      result = a1 + 16 * v20;
      v21 = *(_DWORD *)result <= v7;
      if ( *(_DWORD *)result >= v7 )
      {
        if ( *(_DWORD *)result != v7 || (v21 = *(_DWORD *)(result + 4) <= v11, *(_DWORD *)(result + 4) >= v11) )
        {
          if ( !v21 || *(_QWORD *)(result + 8) >= a5 )
            break;
        }
      }
      v22 = (_QWORD *)(a1 + 16 * v13);
      *v22 = *(_QWORD *)result;
      v22[1] = *(_QWORD *)(result + 8);
      v13 = v20;
      if ( a2 >= v20 )
        goto LABEL_18;
      v20 = (v20 - 1) / 2;
    }
    result = a1 + 16 * v13;
  }
LABEL_18:
  *(_DWORD *)result = v6;
  *(_DWORD *)(result + 4) = v10;
  *(_QWORD *)(result + 8) = a5;
  return result;
}
