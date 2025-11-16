// Function: sub_38BD860
// Address: 0x38bd860
//
void __fastcall sub_38BD860(__int64 a1)
{
  _BYTE *v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // r12
  int v4; // r9d
  __int64 v5; // r8
  unsigned __int64 v6; // rbx
  _QWORD *v7; // r13
  unsigned __int64 v8; // r14
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // rcx
  __m128i *v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r15
  unsigned __int64 v15; // r9
  __int64 v16; // rsi
  unsigned __int64 v17; // r14
  void *v18; // rdi
  __int64 v19; // [rsp+18h] [rbp-C8h]
  __int64 v21; // [rsp+28h] [rbp-B8h]
  __int64 v22; // [rsp+28h] [rbp-B8h]
  __int64 v23; // [rsp+28h] [rbp-B8h]
  __m128i *v24; // [rsp+28h] [rbp-B8h]
  __int64 v25; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v26; // [rsp+38h] [rbp-A8h]
  __int16 v27; // [rsp+40h] [rbp-A0h]
  __m128i s1; // [rsp+50h] [rbp-90h] BYREF
  _BYTE v29[16]; // [rsp+60h] [rbp-80h] BYREF
  __m128i v30; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v31[2]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v32[2]; // [rsp+90h] [rbp-50h] BYREF
  _QWORD v33[8]; // [rsp+A0h] [rbp-40h] BYREF

  v1 = *(_BYTE **)(a1 + 752);
  if ( !v1 )
  {
    v5 = *(_QWORD *)(a1 + 920);
    v7 = v29;
    v6 = 0;
    v3 = a1 + 904;
    s1.m128i_i64[0] = (__int64)v29;
    s1.m128i_i64[1] = 0;
    v29[0] = 0;
    if ( v5 == a1 + 904 )
      goto LABEL_14;
    goto LABEL_3;
  }
  v2 = *(unsigned int *)(a1 + 760);
  s1.m128i_i64[0] = (__int64)v29;
  v3 = a1 + 904;
  sub_38BB9D0(s1.m128i_i64, v1, (__int64)&v1[v2]);
  v5 = *(_QWORD *)(a1 + 920);
  v6 = s1.m128i_u64[1];
  v7 = (_QWORD *)s1.m128i_i64[0];
  if ( v5 != a1 + 904 )
  {
    do
    {
LABEL_3:
      v8 = *(_QWORD *)(v5 + 40);
      if ( v8 <= v6 )
      {
        if ( !v8 || (v21 = v5, v9 = memcmp(v7, *(const void **)(v5 + 32), *(_QWORD *)(v5 + 40)), v5 = v21, !v9) )
        {
          v22 = v5;
          v32[0] = (__int64)v33;
          sub_38BB9D0(v32, (_BYTE *)v7 + v8, (__int64)v7 + v6);
          v26 = v32;
          v25 = v22 + 64;
          v27 = 1028;
          sub_16E2FC0(v30.m128i_i64, (__int64)&v25);
          v10 = v22;
          if ( (_QWORD *)v32[0] != v33 )
          {
            j_j___libc_free_0(v32[0]);
            v10 = v22;
          }
          v23 = v10;
          sub_22415E0(&s1, &v30);
          v5 = v23;
          if ( (_QWORD *)v30.m128i_i64[0] != v31 )
          {
            j_j___libc_free_0(v30.m128i_u64[0]);
            v5 = v23;
          }
          v6 = s1.m128i_u64[1];
          v7 = (_QWORD *)s1.m128i_i64[0];
        }
      }
      v5 = sub_220EF30(v5);
    }
    while ( v5 != v3 );
  }
  *(_DWORD *)(a1 + 760) = 0;
  if ( *(unsigned int *)(a1 + 764) < v6 )
  {
    sub_16CD150(a1 + 752, (const void *)(a1 + 768), v6, 1, v5, v4);
    v18 = (void *)(*(_QWORD *)(a1 + 752) + *(unsigned int *)(a1 + 760));
  }
  else
  {
    if ( !v6 )
      goto LABEL_14;
    v18 = *(void **)(a1 + 752);
  }
  memcpy(v18, v7, v6);
  LODWORD(v6) = *(_DWORD *)(a1 + 760) + v6;
LABEL_14:
  v11 = *(_QWORD *)(a1 + 1000);
  *(_DWORD *)(a1 + 760) = v6;
  v19 = v11;
  if ( a1 + 984 != v11 )
  {
    while ( 1 )
    {
      v12 = *(__m128i **)(v19 + 48);
      v13 = 2LL * *(unsigned int *)(v19 + 56);
      v24 = &v12[v13];
      if ( &v12[v13] != v12 )
        break;
LABEL_29:
      v19 = sub_220EEE0(v19);
      if ( a1 + 984 == v19 )
        goto LABEL_30;
    }
    while ( 1 )
    {
      v14 = *(_QWORD *)(a1 + 920);
      if ( v14 != v3 )
        break;
LABEL_28:
      v12 += 2;
      if ( v24 == v12 )
        goto LABEL_29;
    }
    while ( 1 )
    {
      v17 = *(_QWORD *)(v14 + 40);
      if ( v12->m128i_i64[1] >= v17 )
      {
        if ( !v17 )
        {
          v32[0] = (__int64)v33;
          v16 = v12->m128i_i64[0];
          v15 = v12->m128i_u64[1];
LABEL_20:
          sub_38BB9D0(v32, (_BYTE *)(v17 + v16), v16 + v15);
          v26 = v32;
          v25 = v14 + 64;
          v27 = 1028;
          sub_16E2FC0(v30.m128i_i64, (__int64)&v25);
          if ( (_QWORD *)v32[0] != v33 )
            j_j___libc_free_0(v32[0]);
          sub_22415E0(v12, &v30);
          if ( (_QWORD *)v30.m128i_i64[0] != v31 )
            j_j___libc_free_0(v30.m128i_u64[0]);
          goto LABEL_24;
        }
        if ( !memcmp((const void *)v12->m128i_i64[0], *(const void **)(v14 + 32), *(_QWORD *)(v14 + 40)) )
        {
          v32[0] = (__int64)v33;
          v15 = v12->m128i_u64[1];
          v16 = v12->m128i_i64[0];
          if ( v17 > v15 )
            sub_222CF80(
              "%s: __pos (which is %zu) > this->size() (which is %zu)",
              "basic_string::basic_string",
              v17,
              v12->m128i_u64[1]);
          goto LABEL_20;
        }
      }
LABEL_24:
      v14 = sub_220EF30(v14);
      if ( v14 == v3 )
        goto LABEL_28;
    }
  }
LABEL_30:
  if ( (_BYTE *)s1.m128i_i64[0] != v29 )
    j_j___libc_free_0(s1.m128i_u64[0]);
}
