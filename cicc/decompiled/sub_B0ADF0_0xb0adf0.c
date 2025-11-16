// Function: sub_B0ADF0
// Address: 0xb0adf0
//
_QWORD *__fastcall sub_B0ADF0(__int64 a1, int a2)
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
  unsigned __int16 v14; // ax
  __int64 v15; // rcx
  unsigned __int8 v16; // al
  __int64 *v17; // rsi
  unsigned __int8 v18; // al
  __int64 v19; // rsi
  unsigned __int8 v20; // al
  __int64 v21; // rcx
  unsigned int v22; // edx
  __int64 *v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdi
  int v26; // r11d
  __int64 *v27; // r8
  __int64 v28; // rdx
  _QWORD *j; // rdx
  __int64 v30; // [rsp+0h] [rbp-80h]
  __int64 *v31; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+18h] [rbp-68h]
  int v33; // [rsp+20h] [rbp-60h] BYREF
  __int64 v34; // [rsp+28h] [rbp-58h] BYREF
  __int64 v35; // [rsp+30h] [rbp-50h] BYREF
  __int8 v36[8]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v37[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v31 = v4;
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
    v30 = 8 * v5;
    v9 = &result[v8];
    for ( i = &v4[v5]; v9 != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    v11 = v31;
    if ( &v31[v5] != v31 )
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
          v32 = *(_QWORD *)(a1 + 8);
          v14 = sub_AF18C0(*v11);
          v15 = v12 - 16;
          v33 = v14;
          v16 = *(_BYTE *)(v12 - 16);
          if ( (v16 & 2) != 0 )
            v17 = *(__int64 **)(v12 - 32);
          else
            v17 = (__int64 *)(v15 - 8LL * ((v16 >> 2) & 0xF));
          v34 = *v17;
          v18 = *(_BYTE *)(v12 - 16);
          if ( (v18 & 2) != 0 )
            v19 = *(_QWORD *)(v12 - 32);
          else
            v19 = v15 - 8LL * ((v18 >> 2) & 0xF);
          v35 = *(_QWORD *)(v19 + 8);
          v36[0] = *(_BYTE *)(v12 + 1) >> 7;
          v20 = *(_BYTE *)(v12 - 16);
          if ( (v20 & 2) != 0 )
            v21 = *(_QWORD *)(v12 - 32);
          else
            v21 = v15 - 8LL * ((v20 >> 2) & 0xF);
          v37[0] = *(_QWORD *)(v21 + 16);
          v22 = (v13 - 1) & sub_AF9230(&v33, &v34, &v35, v36, v37);
          v23 = (__int64 *)(v32 + 8LL * v22);
          v24 = *v11;
          v25 = *v23;
          if ( *v23 != *v11 )
          {
            v26 = 1;
            v27 = 0;
            while ( v25 != -4096 )
            {
              if ( v25 != -8192 || v27 )
                v23 = v27;
              v22 = (v13 - 1) & (v26 + v22);
              v25 = *(_QWORD *)(v32 + 8LL * v22);
              if ( v25 == v24 )
              {
                v23 = (__int64 *)(v32 + 8LL * v22);
                goto LABEL_27;
              }
              ++v26;
              v27 = v23;
              v23 = (__int64 *)(v32 + 8LL * v22);
            }
            if ( v27 )
              v23 = v27;
          }
LABEL_27:
          *v23 = v24;
          ++*(_DWORD *)(a1 + 16);
        }
        ++v11;
      }
      while ( i != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v31, v30, 8);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v28]; j != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
