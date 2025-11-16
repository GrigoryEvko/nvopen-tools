// Function: sub_24ABCF0
// Address: 0x24abcf0
//
void __fastcall sub_24ABCF0(__int64 a1)
{
  __int64 v2; // rdx
  char *v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r9
  unsigned __int8 *v8; // rdi
  __int64 v9; // rdx
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _BYTE *v15; // rdi
  size_t v16; // r8
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  _QWORD *v28; // r13
  _QWORD *i; // rbx
  size_t v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // [rsp+28h] [rbp-158h]
  __int64 v36; // [rsp+30h] [rbp-150h]
  __int64 v37[2]; // [rsp+40h] [rbp-140h] BYREF
  _BYTE v38[16]; // [rsp+50h] [rbp-130h] BYREF
  void *v39[2]; // [rsp+60h] [rbp-120h] BYREF
  __int64 v40; // [rsp+70h] [rbp-110h] BYREF
  _QWORD *v41[2]; // [rsp+80h] [rbp-100h] BYREF
  _QWORD v42[2]; // [rsp+90h] [rbp-F0h] BYREF
  _QWORD *v43; // [rsp+A0h] [rbp-E0h] BYREF
  size_t n; // [rsp+A8h] [rbp-D8h]
  _QWORD v45[2]; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i v46; // [rsp+C0h] [rbp-C0h] BYREF
  char *v47; // [rsp+D0h] [rbp-B0h]
  __int16 v48; // [rsp+E0h] [rbp-A0h]
  __m128i v49[2]; // [rsp+F0h] [rbp-90h] BYREF
  __int16 v50; // [rsp+110h] [rbp-70h]
  __m128i v51[2]; // [rsp+120h] [rbp-60h] BYREF
  __int16 v52; // [rsp+140h] [rbp-40h]

  if ( !(unsigned __int8)sub_24ABC70(*(_QWORD *)a1, *(_QWORD **)(a1 + 16)) )
    return;
  v3 = (char *)sub_BD5D20(*(_QWORD *)a1);
  if ( v3 )
  {
    v37[0] = (__int64)v38;
    sub_24A2F70(v37, v3, (__int64)&v3[v2]);
  }
  else
  {
    v38[0] = 0;
    v37[0] = (__int64)v38;
    v37[1] = 0;
  }
  v50 = 267;
  v4 = *(_QWORD *)a1;
  v49[0].m128i_i64[0] = a1 + 200;
  v46.m128i_i64[0] = (__int64)sub_BD5D20(v4);
  v46.m128i_i64[1] = v5;
  v48 = 773;
  v47 = ".";
  sub_9C6370(v51, &v46, v49, v6, 773, v7);
  sub_CA0F50((__int64 *)v39, (void **)v51);
  v8 = *(unsigned __int8 **)a1;
  v52 = 260;
  v51[0].m128i_i64[0] = (__int64)v39;
  sub_BD6B50(v8, (const char **)v51);
  v9 = *(_QWORD *)a1;
  v52 = 260;
  v51[0].m128i_i64[0] = (__int64)v37;
  sub_B305A0(4, (__int64)v51, v9);
  v10 = *(_BYTE **)(a1 + 128);
  v11 = *(_QWORD *)(a1 + 136);
  v46.m128i_i64[0] = a1 + 200;
  v41[0] = v42;
  v48 = 267;
  sub_24A3020((__int64 *)v41, v10, (__int64)&v10[v11]);
  if ( v41[1] == (_QWORD *)0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)v41, ".", 1u);
  v50 = 260;
  v49[0].m128i_i64[0] = (__int64)v41;
  sub_9C6370(v51, v49, &v46, v12, v13, v14);
  sub_CA0F50((__int64 *)&v43, (void **)v51);
  v15 = *(_BYTE **)(a1 + 128);
  if ( v43 == v45 )
  {
    v30 = n;
    if ( n )
    {
      if ( n == 1 )
        *v15 = v45[0];
      else
        memcpy(v15, v45, n);
      v30 = n;
      v15 = *(_BYTE **)(a1 + 128);
    }
    *(_QWORD *)(a1 + 136) = v30;
    v15[v30] = 0;
    v15 = v43;
    goto LABEL_10;
  }
  v16 = n;
  v17 = v45[0];
  if ( v15 == (_BYTE *)(a1 + 144) )
  {
    *(_QWORD *)(a1 + 128) = v43;
    *(_QWORD *)(a1 + 136) = v16;
    *(_QWORD *)(a1 + 144) = v17;
    goto LABEL_30;
  }
  v18 = *(_QWORD *)(a1 + 144);
  *(_QWORD *)(a1 + 128) = v43;
  *(_QWORD *)(a1 + 136) = v16;
  *(_QWORD *)(a1 + 144) = v17;
  if ( !v15 )
  {
LABEL_30:
    v43 = v45;
    v15 = v45;
    goto LABEL_10;
  }
  v43 = v15;
  v45[0] = v18;
LABEL_10:
  n = 0;
  *v15 = 0;
  if ( v43 != v45 )
    j_j___libc_free_0((unsigned __int64)v43);
  if ( v41[0] != v42 )
    j_j___libc_free_0((unsigned __int64)v41[0]);
  v19 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
  v20 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
  if ( v19 )
  {
    v41[0] = *(_QWORD **)(*(_QWORD *)a1 + 48LL);
    v50 = 267;
    v36 = v19;
    v35 = v20;
    v49[0].m128i_i64[0] = a1 + 200;
    v46.m128i_i64[0] = sub_AA8810(v41[0]);
    v46.m128i_i64[1] = v21;
    v48 = 773;
    v47 = ".";
    sub_9C6370(v51, &v46, v49, 773, v22, v23);
    sub_CA0F50((__int64 *)&v43, (void **)v51);
    v24 = sub_BAA410(v35, v43, n);
    *(_DWORD *)(v24 + 8) = *(_DWORD *)(v36 + 8);
    v25 = sub_24ABBE0(*(_QWORD **)(a1 + 16), (unsigned __int64 *)v41);
    v28 = (_QWORD *)v26;
    for ( i = (_QWORD *)v25; v28 != i; i = (_QWORD *)*i )
      sub_B2F990(i[2], v24, v26, v27);
    if ( v43 != v45 )
      j_j___libc_free_0((unsigned __int64)v43);
    if ( v39[0] != &v40 )
      j_j___libc_free_0((unsigned __int64)v39[0]);
    if ( (_BYTE *)v37[0] != v38 )
      j_j___libc_free_0(v37[0]);
  }
  else
  {
    v31 = sub_BAA410(*(_QWORD *)(*(_QWORD *)a1 + 40LL), v39[0], (size_t)v39[1]);
    v33 = *(_QWORD *)a1;
    v34 = v31;
    LOBYTE(v31) = *(_BYTE *)(*(_QWORD *)a1 + 32LL) & 0xF0 | 3;
    *(_BYTE *)(*(_QWORD *)a1 + 32LL) = v31;
    if ( (v31 & 0x30) != 0 )
      *(_BYTE *)(v33 + 33) |= 0x40u;
    sub_B2F990(*(_QWORD *)a1, v34, v33, v32);
    sub_2240A30((unsigned __int64 *)v39);
    sub_2240A30((unsigned __int64 *)v37);
  }
}
