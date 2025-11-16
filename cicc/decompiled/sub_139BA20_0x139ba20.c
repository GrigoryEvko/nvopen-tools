// Function: sub_139BA20
// Address: 0x139ba20
//
__int64 __fastcall sub_139BA20(_QWORD *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rcx
  _BYTE *v6; // r13
  size_t v7; // r12
  const char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rax
  _DWORD *v17; // rdx
  const char *v18; // rsi
  _QWORD *v19; // rdi
  __m128i *v20; // rdx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v24; // rax
  __m128i si128; // xmm0
  char *v26; // rdi
  __int64 v27; // [rsp-F0h] [rbp-F0h] BYREF
  unsigned int v28; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v29; // [rsp-E0h] [rbp-E0h]
  __int64 *v30; // [rsp-D8h] [rbp-D8h] BYREF
  __int16 v31; // [rsp-C8h] [rbp-C8h]
  char *v32; // [rsp-B8h] [rbp-B8h] BYREF
  size_t v33; // [rsp-B0h] [rbp-B0h]
  _QWORD v34[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v35[2]; // [rsp-98h] [rbp-98h] BYREF
  _QWORD v36[2]; // [rsp-88h] [rbp-88h] BYREF
  _QWORD v37[15]; // [rsp-78h] [rbp-78h] BYREF

  v1 = (__int64 *)a1[1];
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F98A8D )
  {
    v2 += 16;
    if ( v3 == v2 )
      BUG();
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F98A8D);
  v6 = (_BYTE *)a1[20];
  v7 = a1[21];
  v32 = (char *)v34;
  v27 = *(_QWORD *)(v4 + 160);
  if ( &v6[v7] && !v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v37[0] = v7;
  if ( v7 > 0xF )
  {
    v32 = (char *)sub_22409D0(&v32, v37, 0);
    v26 = v32;
    v34[0] = v37[0];
  }
  else
  {
    if ( v7 == 1 )
    {
      LOBYTE(v34[0]) = *v6;
      v8 = (const char *)v34;
      goto LABEL_10;
    }
    if ( !v7 )
    {
      v8 = (const char *)v34;
      goto LABEL_10;
    }
    v26 = (char *)v34;
  }
  memcpy(v26, v6, v7);
  v7 = v37[0];
  v8 = v32;
LABEL_10:
  v33 = v7;
  v8[v7] = 0;
  if ( 0x3FFFFFFFFFFFFFFFLL - v33 <= 3 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v32, ".dot", 4, v5);
  v28 = 0;
  v29 = sub_2241E40(&v32, ".dot", v9, v10, v11);
  v13 = sub_16E8CB0(&v32, ".dot", v12);
  v14 = *(_QWORD *)(v13 + 24);
  v15 = v13;
  if ( (unsigned __int64)(*(_QWORD *)(v13 + 16) - v14) <= 8 )
  {
    v15 = sub_16E7EE0(v13, "Writing '", 9);
  }
  else
  {
    *(_BYTE *)(v14 + 8) = 39;
    *(_QWORD *)v14 = 0x20676E6974697257LL;
    *(_QWORD *)(v13 + 24) += 9LL;
  }
  v16 = sub_16E7EE0(v15, v32, v33);
  v17 = *(_DWORD **)(v16 + 24);
  if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 3u )
  {
    sub_16E7EE0(v16, "'...", 4);
  }
  else
  {
    *v17 = 774778407;
    *(_QWORD *)(v16 + 24) += 4LL;
  }
  sub_16E8AF0(v37, v32, v33, &v28, 1);
  v18 = "Call graph";
  v35[0] = (__int64)v36;
  sub_1399600(v35, "Call graph", (__int64)"");
  if ( v28 )
  {
    v24 = sub_16E8CB0(v35, v18, v28);
    v20 = *(__m128i **)(v24 + 24);
    v19 = (_QWORD *)v24;
    if ( *(_QWORD *)(v24 + 16) - (_QWORD)v20 <= 0x20u )
    {
      v18 = "  error opening file for writing!";
      sub_16E7EE0(v24, "  error opening file for writing!", 33);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v20[2].m128i_i8[0] = 33;
      *v20 = si128;
      v20[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      *(_QWORD *)(v24 + 24) += 33LL;
    }
  }
  else
  {
    v19 = v37;
    v18 = (const char *)&v27;
    v31 = 260;
    v30 = v35;
    sub_139B410((__int64)v37, (__int64)&v27, 1, (__int64)&v30);
  }
  v21 = sub_16E8CB0(v19, v18, v20);
  v22 = *(_BYTE **)(v21 + 24);
  if ( *(_BYTE **)(v21 + 16) == v22 )
  {
    sub_16E7EE0(v21, "\n", 1);
  }
  else
  {
    *v22 = 10;
    ++*(_QWORD *)(v21 + 24);
  }
  if ( (_QWORD *)v35[0] != v36 )
    j_j___libc_free_0(v35[0], v36[0] + 1LL);
  sub_16E7C30(v37);
  if ( v32 != (char *)v34 )
    j_j___libc_free_0(v32, v34[0] + 1LL);
  return 0;
}
