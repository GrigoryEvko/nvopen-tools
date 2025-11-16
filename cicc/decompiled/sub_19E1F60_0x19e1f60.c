// Function: sub_19E1F60
// Address: 0x19e1f60
//
__int64 __fastcall sub_19E1F60(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 result; // rax
  unsigned int v6; // r12d
  int v7; // edx
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // r10d
  __int64 v13; // rsi
  unsigned int v14; // ecx
  __int64 v15; // r15
  int v16; // edx
  _QWORD *v17; // r11
  unsigned int v18; // r14d
  int v19; // eax
  int v20; // r10d

  v3 = *a1;
  v4 = a1[1];
  result = (unsigned int)v4 >> 9;
  v6 = result ^ ((unsigned int)v4 >> 4);
  while ( 1 )
  {
    v16 = *(_DWORD *)(a2 + 2384);
    v17 = a1;
    if ( !v16 )
      break;
    v7 = v16 - 1;
    v8 = *(_QWORD *)(a2 + 2368);
    v9 = v7 & v6;
    v10 = (__int64 *)(v8 + 16LL * (v7 & v6));
    v11 = *v10;
    if ( *v10 == v4 )
    {
LABEL_3:
      v12 = *((_DWORD *)v10 + 2);
      v13 = *(a1 - 1);
    }
    else
    {
      v19 = 1;
      while ( v11 != -8 )
      {
        v20 = v19 + 1;
        v9 = v7 & (v19 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( *v10 == v4 )
          goto LABEL_3;
        v19 = v20;
      }
      v13 = *(a1 - 1);
      v12 = 0;
    }
    v14 = v7 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    result = v8 + 16LL * v14;
    v15 = *(_QWORD *)result;
    if ( *(_QWORD *)result != v13 )
    {
      result = 1;
      while ( v15 != -8 )
      {
        v18 = result + 1;
        v14 = v7 & (result + v14);
        result = v8 + 16LL * v14;
        v15 = *(_QWORD *)result;
        if ( *(_QWORD *)result == v13 )
          goto LABEL_5;
        result = v18;
      }
      break;
    }
LABEL_5:
    a1 -= 2;
    if ( *(_DWORD *)(result + 8) <= v12 )
      break;
    a1[2] = *a1;
    result = a1[1];
    a1[3] = result;
  }
  *v17 = v3;
  v17[1] = v4;
  return result;
}
