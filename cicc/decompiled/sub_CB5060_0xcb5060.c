// Function: sub_CB5060
// Address: 0xcb5060
//
__int64 __fastcall sub_CB5060(__int64 a1, char *a2, __int64 a3, _BYTE *a4, _QWORD *a5)
{
  __int64 v6; // r13
  int v7; // eax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r9
  int v12; // esi
  __int64 v13; // rdx
  __m128i *v14; // rcx
  __m128i *v15; // rax
  __int64 v16; // rsi
  size_t v17; // r15
  int v18; // eax
  unsigned int v19; // r10d
  _QWORD *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v24; // rax
  unsigned int v25; // r10d
  _QWORD *v26; // rcx
  _QWORD *v27; // r11
  __int64 *v28; // rax
  __int64 *v29; // rax
  __int64 v30; // rax
  bool v31; // zf
  _QWORD *v32; // rdx
  char v33; // al
  __int64 *v34; // rsi
  unsigned __int64 v35; // r10
  __int64 v36; // rdi
  __int64 v37; // [rsp+8h] [rbp-C8h]
  _QWORD *v38; // [rsp+10h] [rbp-C0h]
  _QWORD *v39; // [rsp+18h] [rbp-B8h]
  __int64 *v40; // [rsp+20h] [rbp-B0h]
  __int64 v41; // [rsp+20h] [rbp-B0h]
  unsigned int v44; // [rsp+38h] [rbp-98h]
  char v45; // [rsp+3Ch] [rbp-94h]
  _QWORD v46[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v47; // [rsp+60h] [rbp-70h]
  __int64 v48[2]; // [rsp+70h] [rbp-60h] BYREF
  _QWORD v49[2]; // [rsp+80h] [rbp-50h] BYREF
  char v50; // [rsp+90h] [rbp-40h]
  char v51; // [rsp+91h] [rbp-3Fh]

  v6 = *(_QWORD *)(a1 + 672);
  v45 = a3;
  if ( !v6 )
  {
    if ( (_BYTE)a3 )
    {
      v30 = sub_2241E50(a1, a2, a3, a4, a5);
      *(_DWORD *)(a1 + 96) = 22;
      *(_QWORD *)(a1 + 104) = v30;
      return 0;
    }
LABEL_21:
    *a4 = 1;
    return 0;
  }
  v7 = *(_DWORD *)(*(_QWORD *)v6 + 32LL);
  if ( v7 != 4 )
  {
    if ( v7 || (_BYTE)a3 )
    {
      v51 = 1;
      v50 = 3;
      v48[0] = (__int64)"not a mapping";
      sub_CB1040(a1, (__int64 *)v6, (__int64)v48);
      return 0;
    }
    goto LABEL_21;
  }
  v9 = -1;
  v48[0] = (__int64)v49;
  if ( a2 )
    v9 = (__int64)&a2[strlen(a2)];
  sub_CB0250(v48, a2, v9);
  v10 = *(unsigned int *)(v6 + 40);
  v11 = v10 + 1;
  v12 = *(_DWORD *)(v6 + 40);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 44) )
  {
    v35 = *(_QWORD *)(v6 + 32);
    v36 = v6 + 32;
    if ( v35 > (unsigned __int64)v48 || (unsigned __int64)v48 >= v35 + 32 * v10 )
    {
      sub_95D880(v36, v11);
      v10 = *(unsigned int *)(v6 + 40);
      v13 = *(_QWORD *)(v6 + 32);
      v14 = (__m128i *)v48;
      v12 = *(_DWORD *)(v6 + 40);
    }
    else
    {
      v41 = *(_QWORD *)(v6 + 32);
      sub_95D880(v36, v11);
      v13 = *(_QWORD *)(v6 + 32);
      v10 = *(unsigned int *)(v6 + 40);
      v12 = *(_DWORD *)(v6 + 40);
      v14 = (__m128i *)((char *)v48 + v13 - v41);
    }
  }
  else
  {
    v13 = *(_QWORD *)(v6 + 32);
    v14 = (__m128i *)v48;
  }
  v15 = (__m128i *)(v13 + 32 * v10);
  if ( v15 )
  {
    v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
    if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
    {
      v15[1] = _mm_loadu_si128(v14 + 1);
    }
    else
    {
      v15->m128i_i64[0] = v14->m128i_i64[0];
      v15[1].m128i_i64[0] = v14[1].m128i_i64[0];
    }
    v16 = v14->m128i_i64[1];
    v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
    v14->m128i_i64[1] = 0;
    v15->m128i_i64[1] = v16;
    v14[1].m128i_i8[0] = 0;
    v12 = *(_DWORD *)(v6 + 40);
  }
  *(_DWORD *)(v6 + 40) = v12 + 1;
  if ( (_QWORD *)v48[0] != v49 )
    j_j___libc_free_0(v48[0], v49[0] + 1LL);
  v17 = 0;
  v40 = (__int64 *)(v6 + 8);
  if ( a2 )
    v17 = strlen(a2);
  v18 = sub_C92610();
  v19 = sub_C92740((__int64)v40, a2, v17, v18);
  v20 = (_QWORD *)(*(_QWORD *)(v6 + 8) + 8LL * v19);
  v21 = *v20;
  if ( *v20 )
  {
    if ( v21 != -8 )
      goto LABEL_17;
    --*(_DWORD *)(v6 + 24);
  }
  v39 = v20;
  v44 = v19;
  v24 = sub_C7D670(v17 + 33, 8);
  v25 = v44;
  v26 = v39;
  v27 = (_QWORD *)v24;
  if ( v17 )
  {
    v38 = (_QWORD *)v24;
    memcpy((void *)(v24 + 32), a2, v17);
    v25 = v44;
    v26 = v39;
    v27 = v38;
  }
  *((_BYTE *)v27 + v17 + 32) = 0;
  *v27 = v17;
  v27[1] = 0;
  v27[2] = 0;
  v27[3] = 0;
  *v26 = v27;
  ++*(_DWORD *)(v6 + 20);
  v28 = (__int64 *)(*(_QWORD *)(v6 + 8) + 8LL * (unsigned int)sub_C929D0(v40, v25));
  v21 = *v28;
  if ( !*v28 || v21 == -8 )
  {
    v29 = v28 + 1;
    do
    {
      do
        v21 = *v29++;
      while ( v21 == -8 );
    }
    while ( !v21 );
  }
LABEL_17:
  v22 = *(_QWORD *)(v21 + 8);
  if ( v22 )
  {
    *a5 = *(_QWORD *)(a1 + 672);
    *(_QWORD *)(a1 + 672) = v22;
    return 1;
  }
  if ( !v45 )
    goto LABEL_21;
  v31 = *a2 == 0;
  v46[0] = "missing required key '";
  if ( v31 )
  {
    v47 = 259;
  }
  else
  {
    v46[2] = a2;
    v47 = 771;
  }
  v32 = v46;
  v33 = 2;
  if ( HIBYTE(v47) == 1 )
  {
    v32 = (_QWORD *)v46[0];
    v37 = v46[1];
    v33 = 3;
  }
  v34 = *(__int64 **)(a1 + 672);
  v48[0] = (__int64)v32;
  v50 = v33;
  v48[1] = v37;
  v49[0] = "'";
  v51 = 3;
  sub_CB1040(a1, v34, (__int64)v48);
  return 0;
}
