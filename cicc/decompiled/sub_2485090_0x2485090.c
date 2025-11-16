// Function: sub_2485090
// Address: 0x2485090
//
__int64 __fastcall sub_2485090(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // r14
  char v7; // r15
  __int64 v8; // rax
  unsigned __int8 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rax
  size_t v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // r10
  int v17; // eax
  const char *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // eax
  int v22; // esi
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v26; // [rsp+10h] [rbp-70h]
  size_t v27; // [rsp+18h] [rbp-68h]
  _QWORD *v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+28h] [rbp-58h]
  void *s2; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD v33[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( a3 == a2 )
    goto LABEL_11;
  v3 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 == 61 )
  {
    if ( !byte_4FE9F68 )
      goto LABEL_11;
    v4 = *(_QWORD *)(a3 - 32);
    v5 = *(_QWORD *)(a3 + 8);
    if ( !v4 )
      goto LABEL_11;
    v6 = 0;
    v7 = 0;
    goto LABEL_6;
  }
  switch ( v3 )
  {
    case '>':
      if ( !(_BYTE)qword_4FE9E88 )
        goto LABEL_11;
      v4 = *(_QWORD *)(a3 - 32);
      v5 = *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL);
      if ( !v4 )
        goto LABEL_11;
      goto LABEL_16;
    case 'B':
      if ( !byte_4FE9DA8 )
        goto LABEL_11;
      v4 = *(_QWORD *)(a3 - 64);
      v5 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 8LL);
      if ( !v4 )
        goto LABEL_11;
      goto LABEL_16;
    case 'A':
      if ( !byte_4FE9DA8 )
        goto LABEL_11;
      v4 = *(_QWORD *)(a3 - 96);
      v5 = *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL);
      if ( !v4 )
        goto LABEL_11;
LABEL_16:
      v6 = 0;
      v7 = 1;
LABEL_6:
      v8 = *(_QWORD *)(v4 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
      {
        v8 = **(_QWORD **)(v8 + 16);
        if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
          v8 = **(_QWORD **)(v8 + 16);
      }
      if ( *(_DWORD *)(v8 + 8) >> 8 || (unsigned __int8)sub_BD6020(v4) )
        goto LABEL_11;
      v10 = sub_BD4CB0((unsigned __int8 *)v4, (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_96, (__int64)&s2);
      v12 = (__int64)v10;
      if ( *v10 != 3 )
        goto LABEL_35;
      if ( (v10[35] & 4) != 0 )
      {
        v26 = v10;
        v13 = sub_B31D10((__int64)v10, (__int64)nullsub_96, v11);
        v27 = v14;
        v25 = v13;
        v15 = sub_B43CA0(a3);
        sub_ED12E0((__int64)&s2, 1, *(_DWORD *)(v15 + 284), 0);
        v16 = s2;
        v12 = (__int64)v26;
        if ( n <= v27 )
        {
          if ( !n || (v29 = s2, v17 = memcmp((const void *)(v27 - n + v25), s2, n), v16 = v29, !v17) )
          {
            if ( v16 != v33 )
              j_j___libc_free_0((unsigned __int64)v16);
            goto LABEL_11;
          }
          v12 = (__int64)v26;
        }
        if ( v16 != v33 )
        {
          v30 = v12;
          j_j___libc_free_0((unsigned __int64)v16);
          v12 = v30;
        }
      }
      v18 = sub_BD5D20(v12);
      if ( v19 <= 5 || *(_DWORD *)v18 != 1819041631 || *((_WORD *)v18 + 2) != 28022 )
      {
LABEL_35:
        *(_QWORD *)a1 = v4;
        *(_BYTE *)(a1 + 8) = v7;
        *(_QWORD *)(a1 + 16) = v5;
        *(_QWORD *)(a1 + 24) = v6;
        *(_BYTE *)(a1 + 32) = 1;
        return a1;
      }
      goto LABEL_11;
  }
  if ( v3 != 85 )
    goto LABEL_11;
  v20 = *(_QWORD *)(a3 - 32);
  if ( !v20 )
    goto LABEL_11;
  v7 = *(_BYTE *)v20;
  if ( *(_BYTE *)v20 || *(_QWORD *)(v20 + 24) != *(_QWORD *)(a3 + 80) )
    goto LABEL_11;
  v21 = *(_DWORD *)(v20 + 36);
  if ( v21 == 228 )
  {
    if ( !byte_4FE9F68 )
      goto LABEL_11;
    v5 = *(_QWORD *)(a3 + 8);
    v22 = *(_DWORD *)(a3 + 4);
    v23 = 0;
  }
  else
  {
    if ( v21 != 230 || !(_BYTE)qword_4FE9E88 )
      goto LABEL_11;
    v22 = *(_DWORD *)(a3 + 4);
    v7 = 1;
    v5 = *(_QWORD *)(*(_QWORD *)(a3 - 32LL * (v22 & 0x7FFFFFF)) + 8LL);
    v23 = 1;
  }
  v24 = v22 & 0x7FFFFFF;
  v4 = *(_QWORD *)(a3 + 32 * (v23 - v24));
  v6 = *(_QWORD *)(a3 + 32 * (v23 + 2 - v24));
  if ( v4 )
    goto LABEL_6;
LABEL_11:
  *(_BYTE *)(a1 + 32) = 0;
  return a1;
}
