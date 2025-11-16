// Function: sub_135CEC0
// Address: 0x135cec0
//
__int64 __fastcall sub_135CEC0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __m128i *v11; // rdx
  __int64 v12; // r13
  __m128i si128; // xmm0
  __int64 v14; // rax
  size_t v15; // rdx
  _BYTE *v16; // rdi
  const char *v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rbx
  __int64 i; // r13
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // r13
  _QWORD *v26; // rbx
  _QWORD *j; // r14
  __int64 v28; // rax
  unsigned __int64 *v29; // rbx
  unsigned __int64 *v30; // r14
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // r15
  unsigned __int64 v33; // r12
  __int64 v34; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  size_t v38; // [rsp+8h] [rbp-98h]
  void *v39; // [rsp+10h] [rbp-90h] BYREF
  char v40[16]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v41; // [rsp+28h] [rbp-78h]
  void *v42; // [rsp+40h] [rbp-60h] BYREF
  char v43[16]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v44; // [rsp+58h] [rbp-48h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_61:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F96DB4 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_61;
  }
  v7 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_4F96DB4)
                 + 160);
  v8 = sub_22077B0(72);
  if ( v8 )
  {
    v9 = v8 + 8;
    *(_QWORD *)v8 = v7;
    *(_QWORD *)(v8 + 16) = v8 + 8;
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 8) = (v8 + 8) | 4;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 40) = 0;
    *(_DWORD *)(v8 + 48) = 0;
    *(_DWORD *)(v8 + 56) = 0;
    *(_QWORD *)(v8 + 64) = 0;
  }
  *(_QWORD *)(a1 + 160) = v8;
  v10 = sub_16E8CB0(72, &unk_4F96DB4, v9);
  v11 = *(__m128i **)(v10 + 24);
  v12 = v10;
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v11 <= 0x18u )
  {
    v12 = sub_16E7EE0(v10, "Alias sets for function '", 25);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C310);
    v11[1].m128i_i8[8] = 39;
    v11[1].m128i_i64[0] = 0x206E6F6974636E75LL;
    *v11 = si128;
    *(_QWORD *)(v10 + 24) += 25LL;
  }
  v14 = sub_1649960(a2);
  v16 = *(_BYTE **)(v12 + 24);
  v17 = (const char *)v14;
  v18 = *(_QWORD *)(v12 + 16) - (_QWORD)v16;
  if ( v15 > v18 )
  {
    v36 = sub_16E7EE0(v12, v17);
    v16 = *(_BYTE **)(v36 + 24);
    v12 = v36;
    if ( *(_QWORD *)(v36 + 16) - (_QWORD)v16 > 2u )
    {
LABEL_13:
      v19 = 14887;
      v16[2] = 10;
      *(_WORD *)v16 = 14887;
      *(_QWORD *)(v12 + 24) += 3LL;
      goto LABEL_14;
    }
  }
  else
  {
    if ( v15 )
    {
      v38 = v15;
      memcpy(v16, v17, v15);
      v37 = *(_QWORD *)(v12 + 16);
      v16 = (_BYTE *)(v38 + *(_QWORD *)(v12 + 24));
      *(_QWORD *)(v12 + 24) = v16;
      v18 = v37 - (_QWORD)v16;
    }
    if ( v18 > 2 )
      goto LABEL_13;
  }
  v17 = "':\n";
  v16 = (_BYTE *)v12;
  sub_16E7EE0(v12, "':\n", 3);
LABEL_14:
  v20 = a2 + 72;
  v21 = *(_QWORD *)(a2 + 80);
  if ( v20 != v21 )
  {
    if ( !v21 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v21 + 24);
      if ( i != v21 + 16 )
        break;
      v21 = *(_QWORD *)(v21 + 8);
      if ( v20 == v21 )
        goto LABEL_20;
      if ( !v21 )
        BUG();
    }
    while ( v20 != v21 )
    {
      v17 = (const char *)(i - 24);
      v16 = *(_BYTE **)(a1 + 160);
      if ( !i )
        v17 = 0;
      sub_135CDE0((__int64)v16, (__int64)v17);
      for ( i = *(_QWORD *)(i + 8); i == v21 - 24 + 40; i = *(_QWORD *)(v21 + 24) )
      {
        v21 = *(_QWORD *)(v21 + 8);
        if ( v20 == v21 )
          goto LABEL_20;
        if ( !v21 )
          BUG();
      }
    }
  }
LABEL_20:
  v23 = *(_QWORD *)(a1 + 160);
  v24 = sub_16E8CB0(v16, v17, v19);
  sub_1359660(v23, v24);
  v25 = *(_QWORD *)(a1 + 160);
  if ( v25 )
  {
    sub_1359CD0(*(_QWORD *)(a1 + 160));
    if ( *(_DWORD *)(v25 + 48) )
    {
      sub_1359800(&v39, -8, 0);
      sub_1359800(&v42, -16, 0);
      v26 = *(_QWORD **)(v25 + 32);
      for ( j = &v26[6 * *(unsigned int *)(v25 + 48)]; j != v26; v26 += 6 )
      {
        v28 = v26[3];
        *v26 = &unk_49EE2B0;
        if ( v28 != 0 && v28 != -8 && v28 != -16 )
          sub_1649B30(v26 + 1);
      }
      v42 = &unk_49EE2B0;
      if ( v44 != 0 && v44 != -8 && v44 != -16 )
        sub_1649B30(v43);
      v39 = &unk_49EE2B0;
      if ( v41 != 0 && v41 != -8 && v41 != -16 )
        sub_1649B30(v40);
    }
    j___libc_free_0(*(_QWORD *)(v25 + 32));
    v29 = *(unsigned __int64 **)(v25 + 16);
    while ( (unsigned __int64 *)(v25 + 8) != v29 )
    {
      v30 = v29;
      v29 = (unsigned __int64 *)v29[1];
      v31 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
      *v29 = v31 | *v29 & 7;
      *(_QWORD *)(v31 + 8) = v29;
      v32 = v30[6];
      v33 = v30[5];
      *v30 &= 7u;
      v30[1] = 0;
      if ( v32 != v33 )
      {
        do
        {
          v34 = *(_QWORD *)(v33 + 16);
          if ( v34 != 0 && v34 != -8 && v34 != -16 )
            sub_1649B30(v33);
          v33 += 24LL;
        }
        while ( v32 != v33 );
        v33 = v30[5];
      }
      if ( v33 )
        j_j___libc_free_0(v33, v30[7] - v33);
      j_j___libc_free_0(v30, 72);
    }
    j_j___libc_free_0(v25, 72);
  }
  return 0;
}
