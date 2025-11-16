// Function: sub_1A327C0
// Address: 0x1a327c0
//
__int64 __fastcall sub_1A327C0(__m128i *a1, __int64 a2)
{
  unsigned __int8 *v4; // rsi
  __int64 v5; // rax
  _BYTE *v6; // r15
  unsigned __int8 *v7; // r13
  __m128i v8; // xmm0
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  unsigned __int8 *v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 *v14; // rax
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  unsigned __int8 *v18; // rdi
  __int64 v19; // rcx
  unsigned __int8 **v20; // rax
  unsigned __int8 **v21; // rdx
  unsigned __int8 *v22; // r8
  unsigned __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // r9
  __int64 v26; // rbx
  char v27; // dl
  __int64 v28; // rcx
  int v29; // esi
  unsigned int v30; // eax
  __int64 v31; // rdi
  unsigned int v33; // esi
  __int64 v34; // rsi
  _QWORD *v35; // rdi
  unsigned int v36; // eax
  unsigned __int8 *v37; // r8
  int v38; // ecx
  unsigned int v39; // edi
  __int64 v40; // rax
  int v41; // r10d
  __int64 v42; // [rsp+0h] [rbp-B0h] BYREF
  unsigned __int8 *v43; // [rsp+8h] [rbp-A8h] BYREF
  unsigned __int8 *v44; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+18h] [rbp-98h]
  __int64 v46; // [rsp+20h] [rbp-90h]
  __int64 v47; // [rsp+28h] [rbp-88h]
  __int64 v48; // [rsp+30h] [rbp-80h]
  __int32 v49; // [rsp+38h] [rbp-78h]
  __m128i v50; // [rsp+40h] [rbp-70h]
  _QWORD v51[2]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v52[10]; // [rsp+60h] [rbp-50h] BYREF

  v4 = (unsigned __int8 *)a1[12].m128i_i64[0];
  v44 = v4;
  if ( v4 )
    sub_1623A60((__int64)&v44, (__int64)v4, 2);
  v5 = a1[12].m128i_i64[1];
  v6 = (_BYTE *)a1[16].m128i_i64[0];
  v7 = (unsigned __int8 *)a1[16].m128i_i64[1];
  v8 = _mm_loadu_si128(a1 + 15);
  v51[0] = v52;
  v45 = v5;
  v9 = a1[13].m128i_i64[0];
  v50 = v8;
  v46 = v9;
  v47 = a1[13].m128i_i64[1];
  v48 = a1[14].m128i_i64[0];
  v49 = a1[14].m128i_i32[2];
  if ( &v6[(_QWORD)v7] && !v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v43 = v7;
  if ( (unsigned __int64)v7 > 0xF )
  {
    v51[0] = sub_22409D0(v51, &v43, 0);
    v35 = (_QWORD *)v51[0];
    v52[0] = v43;
  }
  else
  {
    if ( v7 == (unsigned __int8 *)1 )
    {
      LOBYTE(v52[0]) = *v6;
      v10 = v52;
      goto LABEL_8;
    }
    if ( !v7 )
    {
      v10 = v52;
      goto LABEL_8;
    }
    v35 = v52;
  }
  memcpy(v35, v6, (size_t)v7);
  v7 = v43;
  v10 = (_QWORD *)v51[0];
LABEL_8:
  v51[1] = v7;
  v7[(_QWORD)v10] = 0;
  v11 = a1[10].m128i_i64[1];
  if ( *(_BYTE *)(v11 + 16) != 77 )
  {
    v12 = *(unsigned __int8 **)(v11 + 48);
    v45 = *(_QWORD *)(v11 + 40);
    v46 = v11 + 24;
    v43 = v12;
    if ( v12 )
    {
      sub_1623A60((__int64)&v43, (__int64)v12, 2);
      v13 = v44;
      if ( !v44 )
        goto LABEL_12;
    }
    else
    {
      v13 = v44;
      if ( !v44 )
        goto LABEL_14;
    }
    sub_161E7C0((__int64)&v44, (__int64)v13);
LABEL_12:
    v44 = v43;
    if ( v43 )
      sub_1623210((__int64)&v43, v43, (__int64)&v44);
    goto LABEL_14;
  }
  v34 = sub_157EE30(*(_QWORD *)(v11 + 40));
  if ( v34 )
    v34 -= 24;
  sub_17050D0((__int64 *)&v44, v34);
LABEL_14:
  v14 = (__int64 *)a1[10].m128i_i64[1];
  v15 = (unsigned __int8 *)v14[6];
  v43 = v15;
  if ( !v15 )
  {
    v16 = v44;
    if ( !v44 )
      goto LABEL_20;
    goto LABEL_16;
  }
  sub_1623A60((__int64)&v43, (__int64)v15, 2);
  v16 = v44;
  if ( v44 )
LABEL_16:
    sub_161E7C0((__int64)&v44, (__int64)v16);
  v44 = v43;
  if ( v43 )
    sub_1623210((__int64)&v43, v43, (__int64)&v44);
  v14 = (__int64 *)a1[10].m128i_i64[1];
LABEL_20:
  v17 = sub_1A246E0(a1->m128i_i64, (__int64)&v44, *v14);
  v18 = (unsigned __int8 *)a1[10].m128i_i64[1];
  v19 = v17;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v20 = *(unsigned __int8 ***)(a2 - 8);
    v21 = &v20[3 * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)];
  }
  else
  {
    v20 = (unsigned __int8 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v21 = (unsigned __int8 **)a2;
  }
  if ( v20 != v21 )
  {
    do
    {
      while ( 1 )
      {
        if ( v18 == *v20 )
        {
          if ( v18 )
          {
            v22 = v20[1];
            v23 = (unsigned __int64)v20[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v23 = v22;
            if ( v22 )
              *((_QWORD *)v22 + 2) = *((_QWORD *)v22 + 2) & 3LL | v23;
          }
          *v20 = (unsigned __int8 *)v19;
          if ( v19 )
            break;
        }
        v20 += 3;
        if ( v20 == v21 )
          goto LABEL_33;
      }
      v24 = *(_QWORD *)(v19 + 8);
      v20[1] = (unsigned __int8 *)v24;
      if ( v24 )
        *(_QWORD *)(v24 + 16) = (unsigned __int64)(v20 + 1) | *(_QWORD *)(v24 + 16) & 3LL;
      v20[2] = (unsigned __int8 *)((v19 + 8) | (unsigned __int64)v20[2] & 3);
      *(_QWORD *)(v19 + 8) = v20;
      v20 += 3;
    }
    while ( v20 != v21 );
LABEL_33:
    v18 = (unsigned __int8 *)a1[10].m128i_i64[1];
  }
  v43 = v18;
  if ( (unsigned __int8)sub_1AE9990(v18, 0) )
    sub_1A2EDE0(a1[2].m128i_i64[0] + 208, (__int64 *)&v43);
  sub_1A22950(a1->m128i_i64, a2);
  v26 = a1[11].m128i_i64[0];
  v42 = a2;
  v27 = *(_BYTE *)(v26 + 8) & 1;
  if ( v27 )
  {
    v28 = v26 + 16;
    v29 = 7;
  }
  else
  {
    v33 = *(_DWORD *)(v26 + 24);
    v28 = *(_QWORD *)(v26 + 16);
    if ( !v33 )
    {
      v36 = *(_DWORD *)(v26 + 8);
      ++*(_QWORD *)v26;
      v37 = 0;
      v38 = (v36 >> 1) + 1;
LABEL_59:
      v39 = 3 * v33;
      goto LABEL_60;
    }
    v29 = v33 - 1;
  }
  v30 = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (unsigned __int8 *)(v28 + 8LL * v30);
  v31 = *(_QWORD *)v25;
  if ( a2 == *(_QWORD *)v25 )
    goto LABEL_39;
  v41 = 1;
  v37 = 0;
  while ( v31 != -8 )
  {
    if ( v31 != -16 || v37 )
      v25 = v37;
    v30 = v29 & (v41 + v30);
    v31 = *(_QWORD *)(v28 + 8LL * v30);
    if ( a2 == v31 )
      goto LABEL_39;
    ++v41;
    v37 = v25;
    v25 = (unsigned __int8 *)(v28 + 8LL * v30);
  }
  v36 = *(_DWORD *)(v26 + 8);
  if ( !v37 )
    v37 = v25;
  ++*(_QWORD *)v26;
  v38 = (v36 >> 1) + 1;
  if ( !v27 )
  {
    v33 = *(_DWORD *)(v26 + 24);
    goto LABEL_59;
  }
  v39 = 24;
  v33 = 8;
LABEL_60:
  if ( v39 <= 4 * v38 )
  {
    v33 *= 2;
    goto LABEL_74;
  }
  if ( v33 - *(_DWORD *)(v26 + 12) - v38 <= v33 >> 3 )
  {
LABEL_74:
    sub_1A32450(v26, v33);
    sub_1A275C0(v26, &v42, &v43);
    v37 = v43;
    v36 = *(_DWORD *)(v26 + 8);
  }
  *(_DWORD *)(v26 + 8) = (2 * (v36 >> 1) + 2) | v36 & 1;
  if ( *(_QWORD *)v37 != -8 )
    --*(_DWORD *)(v26 + 12);
  *(_QWORD *)v37 = v42;
  v40 = *(unsigned int *)(v26 + 88);
  if ( (unsigned int)v40 >= *(_DWORD *)(v26 + 92) )
  {
    sub_16CD150(v26 + 80, (const void *)(v26 + 96), 0, 8, (int)v37, (int)v25);
    v40 = *(unsigned int *)(v26 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v26 + 80) + 8 * v40) = v42;
  ++*(_DWORD *)(v26 + 88);
LABEL_39:
  if ( (_QWORD *)v51[0] != v52 )
    j_j___libc_free_0(v51[0], v52[0] + 1LL);
  if ( v44 )
    sub_161E7C0((__int64)&v44, (__int64)v44);
  return 1;
}
