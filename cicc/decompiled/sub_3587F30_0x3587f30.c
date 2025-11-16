// Function: sub_3587F30
// Address: 0x3587f30
//
bool __fastcall sub_3587F30(__int64 a1, __int64 *a2)
{
  __int64 v4; // r13
  __int64 v5; // rsi
  __int64 v6; // rdx
  __m128i *v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  bool result; // al
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 (__fastcall ***v14)(); // rax
  __int64 v15; // r15
  __int64 (__fastcall ***v16)(); // r13
  __int64 (__fastcall ***v17)(); // rdx
  int v18; // r14d
  __int64 v19; // rdi
  bool v20; // al
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned __int64 v24; // r14
  __int64 v25; // [rsp-10h] [rbp-F0h]
  bool v26; // [rsp+Fh] [rbp-D1h]
  __int64 v27; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+18h] [rbp-C8h]
  char v29; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v30[2]; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v31; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 *v32; // [rsp+50h] [rbp-90h] BYREF
  __int16 v33; // [rsp+70h] [rbp-70h]
  unsigned __int64 v34[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v35[2]; // [rsp+90h] [rbp-50h] BYREF
  int v36; // [rsp+A0h] [rbp-40h]
  unsigned __int64 **v37; // [rsp+A8h] [rbp-38h]

  v4 = *a2;
  v5 = *(_QWORD *)(a1 + 1208);
  sub_C25870(
    (__int64)&v27,
    v5,
    *(_QWORD *)(a1 + 1216),
    v4,
    *(_QWORD *)(a1 + 1272),
    *(_DWORD *)(a1 + 1304),
    *(_QWORD *)(a1 + 1240),
    *(_QWORD *)(a1 + 1248));
  v6 = v25;
  if ( (v29 & 1) != 0 )
  {
    v6 = (unsigned int)v27;
    v5 = v28;
    if ( (_DWORD)v27 )
    {
      (*(void (__fastcall **)(unsigned __int64 *))(*(_QWORD *)v28 + 32LL))(v34);
      v7 = (__m128i *)sub_2241130(v34, 0, 0, "Could not open profile: ", 0x18u);
      v30[0] = (unsigned __int64)&v31;
      if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
      {
        v31 = _mm_loadu_si128(v7 + 1);
      }
      else
      {
        v30[0] = v7->m128i_i64[0];
        v31.m128i_i64[0] = v7[1].m128i_i64[0];
      }
      v8 = v7->m128i_u64[1];
      v7[1].m128i_i8[0] = 0;
      v30[1] = v8;
      v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
      v7->m128i_i64[1] = 0;
      if ( (_QWORD *)v34[0] != v35 )
        j_j___libc_free_0(v34[0]);
      v33 = 260;
      v9 = *(_QWORD *)(a1 + 1208);
      v32 = v30;
      v10 = *(_QWORD *)(a1 + 1216);
      v34[1] = 12;
      v35[1] = v10;
      v34[0] = (unsigned __int64)&unk_49D9C78;
      v35[0] = v9;
      v36 = 0;
      v37 = &v32;
      sub_B6EB20(v4, (__int64)v34);
      if ( (__m128i *)v30[0] != &v31 )
        j_j___libc_free_0(v30[0]);
      result = 0;
      goto LABEL_10;
    }
  }
  v12 = v27;
  v13 = *(_QWORD *)(a1 + 1136);
  v27 = 0;
  *(_QWORD *)(a1 + 1136) = v12;
  if ( v13 )
  {
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v13 + 8LL))(v13, v5, v6);
    v12 = *(_QWORD *)(a1 + 1136);
  }
  *(_QWORD *)(v12 + 192) = a2;
  v14 = sub_C1AFD0();
  v15 = *(_QWORD *)(a1 + 1136);
  v16 = v14;
  v18 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 24LL))(v15, v5);
  if ( !v18 )
  {
    v19 = *(_QWORD *)(v15 + 88);
    if ( v19 )
      sub_C2D850(v19, *(_QWORD *)(v15 + 64));
    unk_4F838D1 = *(_BYTE *)(v15 + 204);
    v17 = sub_C1AFD0();
  }
  v20 = 0;
  if ( v16 == v17 )
    v20 = v18 == 0;
  v21 = *(_QWORD *)(a1 + 1136);
  *(_BYTE *)(a1 + 1316) = v20;
  result = 1;
  if ( !*(_BYTE *)(v21 + 177) )
  {
LABEL_10:
    if ( (v29 & 1) != 0 )
      return result;
    goto LABEL_26;
  }
  v22 = sub_22077B0(0x20u);
  v23 = v22;
  if ( v22 )
    sub_26C9BA0(v22, (__int64)a2);
  v24 = *(_QWORD *)(a1 + 1192);
  *(_QWORD *)(a1 + 1192) = v23;
  if ( v24 )
  {
    sub_C7D6A0(*(_QWORD *)(v24 + 8), 24LL * *(unsigned int *)(v24 + 24), 8);
    j_j___libc_free_0(v24);
  }
  result = sub_BA8DC0((__int64)a2, (__int64)"llvm.pseudo_probe_desc", 22) != 0;
  if ( (v29 & 1) == 0 )
  {
LABEL_26:
    if ( v27 )
    {
      v26 = result;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
      return v26;
    }
  }
  return result;
}
