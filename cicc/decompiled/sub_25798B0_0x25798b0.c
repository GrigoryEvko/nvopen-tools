// Function: sub_25798B0
// Address: 0x25798b0
//
__int64 __fastcall sub_25798B0(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax
  __int64 *v3; // r15
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int8 v8; // r10
  __int64 v9; // r11
  __int64 *v10; // rdi
  __int64 *v11; // r8
  __int64 v12; // r9
  int v13; // edx
  int v14; // eax
  int v15; // eax
  int v16; // [rsp+Ch] [rbp-44h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(__int64 **)(a1 + 32);
  result = 3LL * *(unsigned int *)(a1 + 40);
  v3 = &v1[3 * *(unsigned int *)(a1 + 40)];
  if ( v3 == v1 )
    return result;
  do
  {
    v5 = *(_DWORD *)(a1 + 24);
    if ( !v5 )
    {
      ++*(_QWORD *)a1;
      v17[0] = 0;
LABEL_13:
      v5 *= 2;
      goto LABEL_14;
    }
    v6 = *v1;
    v7 = v1[1];
    v16 = 1;
    v8 = *((_BYTE *)v1 + 16);
    v9 = *(_QWORD *)(a1 + 8);
    v10 = 0;
    for ( result = (v5 - 1)
                 & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                  * ((37 * (unsigned int)v8)
                                   | ((((0xBF58476D1CE4E5B9LL
                                       * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
                                        | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
                                     ^ (0xBF58476D1CE4E5B9LL
                                      * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
                                       | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32)))) << 32))) >> 31)
                  ^ (756364221 * v8)); ; result = (v5 - 1) & v14 )
    {
      v11 = (__int64 *)(v9 + 24LL * (unsigned int)result);
      v12 = *v11;
      if ( *v11 == v6 && v7 == v11[1] )
        break;
      if ( v12 == -4096 )
        goto LABEL_23;
LABEL_6:
      if ( v12 == -8192 && v11[1] == -8192 && *((_BYTE *)v11 + 16) == 0xFE && !v10 )
        v10 = (__int64 *)(v9 + 24LL * (unsigned int)result);
LABEL_24:
      v14 = v16 + result;
      ++v16;
    }
    if ( v8 == *((_BYTE *)v11 + 16) )
      goto LABEL_18;
    if ( v12 != -4096 )
      goto LABEL_6;
LABEL_23:
    if ( v11[1] != -4096 || *((_BYTE *)v11 + 16) != 0xFF )
      goto LABEL_24;
    v15 = *(_DWORD *)(a1 + 16);
    if ( !v10 )
      v10 = v11;
    ++*(_QWORD *)a1;
    v13 = v15 + 1;
    v17[0] = v10;
    if ( 4 * (v15 + 1) >= 3 * v5 )
      goto LABEL_13;
    if ( v5 - *(_DWORD *)(a1 + 20) - v13 <= v5 >> 3 )
    {
LABEL_14:
      sub_2579470(a1, v5);
      sub_256E050(a1, v1, v17);
      v10 = (__int64 *)v17[0];
      v13 = *(_DWORD *)(a1 + 16) + 1;
    }
    *(_DWORD *)(a1 + 16) = v13;
    if ( *v10 != -4096 || v10[1] != -4096 || *((_BYTE *)v10 + 16) != 0xFF )
      --*(_DWORD *)(a1 + 20);
    *v10 = *v1;
    v10[1] = v1[1];
    result = *((unsigned __int8 *)v1 + 16);
    *((_BYTE *)v10 + 16) = result;
LABEL_18:
    v1 += 3;
  }
  while ( v3 != v1 );
  return result;
}
