// Function: sub_B049F0
// Address: 0xb049f0
//
_QWORD *__fastcall sub_B049F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 *v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rdx
  __int64 *i; // rbx
  __int64 *v11; // r15
  __int64 v12; // r12
  int v13; // r13d
  unsigned __int8 v14; // al
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rdi
  int v20; // r11d
  __int64 *v21; // r8
  __int64 v22; // rdx
  _QWORD *j; // rdx
  __int64 v24; // [rsp+0h] [rbp-80h]
  __int64 *v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  int v27; // [rsp+20h] [rbp-60h] BYREF
  __int64 v28; // [rsp+28h] [rbp-58h] BYREF
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  int v30; // [rsp+38h] [rbp-48h] BYREF
  int v31[17]; // [rsp+3Ch] [rbp-44h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v25 = v4;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v24 = 8 * v5;
    v9 = &result[v8];
    for ( i = &v4[v5]; v9 != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    v11 = v25;
    if ( &v25[v5] != v25 )
    {
      do
      {
        v12 = *v11;
        if ( *v11 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *v11;
            BUG();
          }
          v26 = *(_QWORD *)(a1 + 8);
          v27 = (unsigned __int16)sub_AF18C0(*v11);
          v14 = *(_BYTE *)(v12 - 16);
          if ( (v14 & 2) != 0 )
            v15 = *(_QWORD *)(v12 - 32);
          else
            v15 = v12 - 16 - 8LL * ((v14 >> 2) & 0xF);
          v28 = *(_QWORD *)(v15 + 16);
          v29 = *(_QWORD *)(v12 + 24);
          v30 = sub_AF18D0(v12);
          v31[0] = *(_DWORD *)(v12 + 44);
          v31[1] = *(_DWORD *)(v12 + 40);
          v31[2] = *(_DWORD *)(v12 + 20);
          v16 = (v13 - 1) & sub_AF9B00(&v27, &v28, &v29, &v30, v31);
          v17 = (__int64 *)(v26 + 8LL * v16);
          v18 = *v11;
          v19 = *v17;
          if ( *v17 != *v11 )
          {
            v20 = 1;
            v21 = 0;
            while ( v19 != -4096 )
            {
              if ( v19 != -8192 || v21 )
                v17 = v21;
              v16 = (v13 - 1) & (v20 + v16);
              v19 = *(_QWORD *)(v26 + 8LL * v16);
              if ( v19 == v18 )
              {
                v17 = (__int64 *)(v26 + 8LL * v16);
                goto LABEL_23;
              }
              ++v20;
              v21 = v17;
              v17 = (__int64 *)(v26 + 8LL * v16);
            }
            if ( v21 )
              v17 = v21;
          }
LABEL_23:
          *v17 = v18;
          ++*(_DWORD *)(a1 + 16);
        }
        ++v11;
      }
      while ( i != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v25, v24, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v22]; j != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
