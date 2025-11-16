// Function: sub_15B7120
// Address: 0x15b7120
//
__int64 __fastcall sub_15B7120(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // rcx
  __int64 v10; // rdx
  int v11; // ebx
  int v12; // eax
  __int64 v13; // rsi
  int v14; // r8d
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  int v19; // [rsp+0h] [rbp-40h] BYREF
  int v20; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  __int64 v22[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v19 = *(_DWORD *)(*a2 + 4);
    v20 = *(unsigned __int16 *)(v6 + 2);
    v9 = *(unsigned int *)(v6 + 8);
    v21 = *(_QWORD *)(v6 - 8 * v9);
    v10 = 0;
    if ( (_DWORD)v9 == 2 )
      v10 = *(_QWORD *)(v6 - 8);
    v22[0] = v10;
    v11 = v4 - 1;
    v12 = sub_15B3100(&v19, &v20, &v21, v22);
    v13 = *a2;
    v14 = 1;
    v15 = 0;
    v16 = v11 & v12;
    v17 = (_QWORD *)(v7 + 8LL * v16);
    v18 = *v17;
    if ( *a2 == *v17 )
    {
LABEL_12:
      *a3 = v17;
      return 1;
    }
    else
    {
      while ( v18 != -8 )
      {
        if ( v18 != -16 || v15 )
          v17 = v15;
        v16 = v11 & (v14 + v16);
        v18 = *(_QWORD *)(v7 + 8LL * v16);
        if ( v18 == v13 )
        {
          v17 = (_QWORD *)(v7 + 8LL * v16);
          goto LABEL_12;
        }
        ++v14;
        v15 = v17;
        v17 = (_QWORD *)(v7 + 8LL * v16);
      }
      if ( !v15 )
        v15 = v17;
      *a3 = v15;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
