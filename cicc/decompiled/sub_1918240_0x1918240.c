// Function: sub_1918240
// Address: 0x1918240
//
__int64 __fastcall sub_1918240(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 result; // rax
  int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // r15
  unsigned int v13; // esi
  __int64 v14; // rdi
  __int64 v15; // r9
  unsigned int v16; // edx
  __int64 v17; // r8
  unsigned int v18; // r8d
  int v19; // ebx
  __int64 v20; // r11
  int v21; // edx
  int v22; // r8d
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h] BYREF
  __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a2;
  v23 = a1 + 112;
  result = *(unsigned int *)(a1 + 136);
  v24 = a2;
  if ( (_DWORD)result )
  {
    v5 = result - 1;
    v6 = *(_QWORD *)(a1 + 120);
    v7 = (result - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    result = v6 + 16LL * v7;
    v8 = *(_QWORD *)result;
    if ( v3 == *(_QWORD *)result )
    {
LABEL_3:
      *(_QWORD *)result = -16;
      v3 = v24;
      --*(_DWORD *)(a1 + 128);
      ++*(_DWORD *)(a1 + 132);
    }
    else
    {
      result = 1;
      while ( v8 != -8 )
      {
        v18 = result + 1;
        v7 = v5 & (result + v7);
        result = v6 + 16LL * v7;
        v8 = *(_QWORD *)result;
        if ( v3 == *(_QWORD *)result )
          goto LABEL_3;
        result = v18;
      }
    }
  }
  v9 = *(_QWORD *)(v3 + 48);
  v10 = v3 + 40;
  if ( v9 != v10 )
  {
    while ( 1 )
    {
      v11 = v9 - 24;
      if ( !v9 )
        v11 = 0;
      v12 = v11;
      result = sub_14AE440(v11);
      if ( !(_BYTE)result )
      {
        result = (unsigned int)*(unsigned __int8 *)(v12 + 16) - 54;
        if ( (unsigned __int8)(*(_BYTE *)(v12 + 16) - 54) > 1u )
          break;
      }
      v9 = *(_QWORD *)(v9 + 8);
      if ( v10 == v9 )
        return result;
    }
    v13 = *(_DWORD *)(a1 + 136);
    if ( v13 )
    {
      v14 = v24;
      v15 = *(_QWORD *)(a1 + 120);
      v16 = (v13 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      result = v15 + 16LL * v16;
      v17 = *(_QWORD *)result;
      if ( *(_QWORD *)result == v24 )
      {
LABEL_11:
        *(_QWORD *)(result + 8) = v12;
        return result;
      }
      v19 = 1;
      v20 = 0;
      while ( v17 != -8 )
      {
        if ( v17 == -16 && !v20 )
          v20 = result;
        v16 = (v13 - 1) & (v19 + v16);
        result = v15 + 16LL * v16;
        v17 = *(_QWORD *)result;
        if ( v24 == *(_QWORD *)result )
          goto LABEL_11;
        ++v19;
      }
      v21 = *(_DWORD *)(a1 + 128);
      if ( v20 )
        result = v20;
      ++*(_QWORD *)(a1 + 112);
      v22 = v21 + 1;
      if ( 4 * (v21 + 1) < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(a1 + 132) - v22 > v13 >> 3 )
        {
LABEL_25:
          *(_DWORD *)(a1 + 128) = v22;
          if ( *(_QWORD *)result != -8 )
            --*(_DWORD *)(a1 + 132);
          *(_QWORD *)(result + 8) = 0;
          *(_QWORD *)result = v14;
          *(_QWORD *)(result + 8) = v12;
          return result;
        }
LABEL_30:
        sub_1918080(v23, v13);
        sub_190CC30(v23, &v24, v25);
        result = v25[0];
        v14 = v24;
        v22 = *(_DWORD *)(a1 + 128) + 1;
        goto LABEL_25;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 112);
    }
    v13 *= 2;
    goto LABEL_30;
  }
  return result;
}
