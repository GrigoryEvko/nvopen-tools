// Function: sub_28C8130
// Address: 0x28c8130
//
__int64 __fastcall sub_28C8130(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 result; // rax
  unsigned int v6; // r12d
  int v7; // edx
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r14
  unsigned int v11; // r10d
  unsigned int v12; // ecx
  __int64 v13; // r15
  int v14; // edx
  __int64 v15; // rsi
  _QWORD *v16; // r11
  __int64 v17; // r8
  unsigned int v18; // r14d
  int v19; // eax
  int v20; // r10d

  v3 = *a1;
  v4 = a1[1];
  result = (unsigned int)v4 >> 9;
  v6 = result ^ ((unsigned int)v4 >> 4);
  while ( 1 )
  {
    v14 = *(_DWORD *)(a2 + 2376);
    v15 = *(a1 - 1);
    v16 = a1;
    v17 = *(_QWORD *)(a2 + 2360);
    if ( !v14 )
      break;
    v7 = v14 - 1;
    v8 = v7 & v6;
    v9 = (__int64 *)(v17 + 16LL * (v7 & v6));
    v10 = *v9;
    if ( *v9 == v4 )
    {
LABEL_3:
      v11 = *((_DWORD *)v9 + 2);
    }
    else
    {
      v19 = 1;
      while ( v10 != -4096 )
      {
        v20 = v19 + 1;
        v8 = v7 & (v19 + v8);
        v9 = (__int64 *)(v17 + 16LL * v8);
        v10 = *v9;
        if ( *v9 == v4 )
          goto LABEL_3;
        v19 = v20;
      }
      v11 = 0;
    }
    v12 = v7 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    result = v17 + 16LL * v12;
    v13 = *(_QWORD *)result;
    if ( v15 != *(_QWORD *)result )
    {
      result = 1;
      while ( v13 != -4096 )
      {
        v18 = result + 1;
        v12 = v7 & (result + v12);
        result = v17 + 16LL * v12;
        v13 = *(_QWORD *)result;
        if ( v15 == *(_QWORD *)result )
          goto LABEL_5;
        result = v18;
      }
      break;
    }
LABEL_5:
    a1 -= 2;
    if ( v11 >= *(_DWORD *)(result + 8) )
      break;
    a1[2] = *a1;
    result = a1[1];
    a1[3] = result;
  }
  *v16 = v3;
  v16[1] = v4;
  return result;
}
