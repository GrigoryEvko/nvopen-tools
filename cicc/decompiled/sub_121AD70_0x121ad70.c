// Function: sub_121AD70
// Address: 0x121ad70
//
__int64 __fastcall sub_121AD70(__int64 a1)
{
  _BYTE *v1; // rsi
  unsigned __int64 v2; // r13
  unsigned int v3; // r14d
  size_t v5; // r14
  int v6; // eax
  unsigned int v7; // r9d
  _QWORD *v8; // rcx
  __int64 v9; // rdx
  int v10; // eax
  unsigned int v11; // r9d
  _QWORD *v12; // r11
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // r9d
  _QWORD *v16; // rcx
  _QWORD *v17; // r11
  __int64 *v18; // rax
  __int64 *v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  size_t v22; // rdx
  unsigned int v23; // r9d
  _QWORD *v24; // r11
  _QWORD *v25; // rcx
  __int64 *v26; // rdx
  __int64 *v27; // rdx
  _QWORD *v28; // [rsp+0h] [rbp-C0h]
  _QWORD *v29; // [rsp+8h] [rbp-B8h]
  _QWORD *v30; // [rsp+8h] [rbp-B8h]
  _QWORD *v31; // [rsp+10h] [rbp-B0h]
  unsigned int v32; // [rsp+10h] [rbp-B0h]
  size_t v33; // [rsp+18h] [rbp-A8h]
  unsigned int v34; // [rsp+18h] [rbp-A8h]
  __int64 *v35; // [rsp+20h] [rbp-A0h]
  void *src; // [rsp+28h] [rbp-98h]
  void *srca; // [rsp+28h] [rbp-98h]
  __int64 *v38; // [rsp+38h] [rbp-88h] BYREF
  _QWORD *v39; // [rsp+40h] [rbp-80h] BYREF
  size_t n; // [rsp+48h] [rbp-78h]
  _QWORD v41[2]; // [rsp+50h] [rbp-70h] BYREF
  const char *v42; // [rsp+60h] [rbp-60h] BYREF
  char v43; // [rsp+80h] [rbp-40h]
  char v44; // [rsp+81h] [rbp-3Fh]

  v1 = *(_BYTE **)(a1 + 248);
  v39 = v41;
  sub_12060D0((__int64 *)&v39, v1, (__int64)&v1[*(_QWORD *)(a1 + 256)]);
  v2 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after name")
    || (unsigned __int8)sub_120AFE0(a1, 292, "expected 'type' after name") )
  {
    goto LABEL_2;
  }
  v5 = n;
  v38 = 0;
  v35 = (__int64 *)(a1 + 928);
  src = v39;
  v6 = sub_C92610();
  v7 = sub_C92740(a1 + 928, src, v5, v6);
  v8 = (_QWORD *)(*(_QWORD *)(a1 + 928) + 8LL * v7);
  v9 = *v8;
  if ( *v8 )
  {
    if ( v9 != -8 )
      goto LABEL_9;
    --*(_DWORD *)(a1 + 944);
  }
  v31 = v8;
  v34 = v7;
  v14 = sub_C7D670(v5 + 25, 8);
  v15 = v34;
  v16 = v31;
  v17 = (_QWORD *)v14;
  if ( v5 )
  {
    v30 = (_QWORD *)v14;
    memcpy((void *)(v14 + 24), src, v5);
    v15 = v34;
    v16 = v31;
    v17 = v30;
  }
  *((_BYTE *)v17 + v5 + 24) = 0;
  *v17 = v5;
  v17[1] = 0;
  v17[2] = 0;
  *v16 = v17;
  ++*(_DWORD *)(a1 + 940);
  v18 = (__int64 *)(*(_QWORD *)(a1 + 928) + 8LL * (unsigned int)sub_C929D0(v35, v15));
  v9 = *v18;
  if ( !*v18 || v9 == -8 )
  {
    v19 = v18 + 1;
    do
    {
      do
        v9 = *v19++;
      while ( v9 == -8 );
    }
    while ( !v9 );
  }
LABEL_9:
  v3 = sub_121A7A0(a1, v2, v39, n, (unsigned __int64 *)(v9 + 8), &v38);
  if ( (_BYTE)v3 )
  {
LABEL_2:
    v3 = 1;
    goto LABEL_3;
  }
  if ( *((_BYTE *)v38 + 8) != 15 )
  {
    v33 = n;
    srca = v39;
    v10 = sub_C92610();
    v11 = sub_C92740((__int64)v35, srca, v33, v10);
    v12 = (_QWORD *)(*(_QWORD *)(a1 + 928) + 8LL * v11);
    v13 = *v12;
    if ( *v12 )
    {
      if ( v13 != -8 )
      {
LABEL_13:
        if ( *(_QWORD *)(v13 + 8) )
        {
          v44 = 1;
          v42 = "non-struct types may not be recursive";
          v43 = 3;
          sub_11FD800(a1 + 176, v2, (__int64)&v42, 1);
          goto LABEL_2;
        }
        v20 = v38;
        *(_QWORD *)(v13 + 16) = 0;
        *(_QWORD *)(v13 + 8) = v20;
        goto LABEL_3;
      }
      --*(_DWORD *)(a1 + 944);
    }
    v29 = v12;
    v32 = v11;
    v21 = sub_C7D670(v33 + 25, 8);
    v22 = v33;
    v23 = v32;
    v24 = v29;
    v25 = (_QWORD *)v21;
    if ( v33 )
    {
      v28 = (_QWORD *)v21;
      memcpy((void *)(v21 + 24), srca, v33);
      v22 = v33;
      v23 = v32;
      v24 = v29;
      v25 = v28;
    }
    *((_BYTE *)v25 + v22 + 24) = 0;
    *v25 = v22;
    v25[1] = 0;
    v25[2] = 0;
    *v24 = v25;
    ++*(_DWORD *)(a1 + 940);
    v26 = (__int64 *)(*(_QWORD *)(a1 + 928) + 8LL * (unsigned int)sub_C929D0(v35, v23));
    v13 = *v26;
    if ( !*v26 || v13 == -8 )
    {
      v27 = v26 + 1;
      do
      {
        do
          v13 = *v27++;
        while ( !v13 );
      }
      while ( v13 == -8 );
    }
    goto LABEL_13;
  }
LABEL_3:
  if ( v39 != v41 )
    j_j___libc_free_0(v39, v41[0] + 1LL);
  return v3;
}
