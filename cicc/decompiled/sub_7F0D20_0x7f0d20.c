// Function: sub_7F0D20
// Address: 0x7f0d20
//
_QWORD *__fastcall sub_7F0D20(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  char v7; // al
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  const __m128i *v13; // r13
  __int64 *v14; // rax
  char v15; // bl
  __int64 *v16; // r14
  unsigned __int8 v17; // dl
  const __m128i *v18; // rdi
  __int64 *v19; // rax
  _BYTE *v20; // rax
  unsigned __int8 v21; // dl
  const __m128i *v22; // rdi
  void *v24; // r14
  _QWORD *v25; // rax
  void *v26; // rax
  _BYTE *v27; // rax
  __int64 *v28; // rax
  _BYTE *v29; // [rsp+8h] [rbp-48h]
  const __m128i *v30; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 72);
  v30 = (const __m128i *)sub_724DC0();
  v7 = *(_BYTE *)(a1 + 56);
  if ( v7 == 37 || v7 == 35 && (*(_BYTE *)(a1 + 25) & 4) != 0 )
  {
    if ( dword_4F077C4 == 2 )
      v21 = unk_4F06B39;
    else
      v21 = unk_4F06B38;
    sub_72BAF0((__int64)v30, 1, v21);
    v22 = v30;
    v30[8].m128i_i64[0] = sub_72C390();
    *(_QWORD *)(v2 + 16) = sub_73A720(v22, 1);
    sub_73D8E0(a1, 0x49u, *(_QWORD *)a1, *(_BYTE *)(a1 + 25) & 1, v2);
  }
  else
  {
    v8 = sub_7E25B0(v2, 0, v3, v4, v5, v6);
    v13 = (const __m128i *)sub_731370(v2, 0, v9, v10, v11, v12);
    v14 = sub_7E8090(v13, 1u);
    v15 = *(_BYTE *)(a1 + 56);
    v16 = v14;
    if ( v15 == 35 )
    {
      if ( dword_4F077C4 == 2 )
        v17 = unk_4F06B39;
      else
        v17 = unk_4F06B38;
      sub_72BAF0((__int64)v30, 1, v17);
      v18 = v30;
      v30[8].m128i_i64[0] = sub_72C390();
      *(_QWORD *)(v8 + 16) = sub_73A720(v18, 1);
    }
    else
    {
      v24 = sub_7F0830(v14);
      v25 = sub_72BA30(5u);
      v26 = sub_73DBF0(0x1Du, (__int64)v25, (__int64)v24);
      v27 = sub_73E110((__int64)v26, v13->m128i_i64[0]);
      if ( v15 == 38 )
      {
        *(_QWORD *)(v8 + 16) = v27;
        v28 = (__int64 *)sub_73DBF0(0x49u, v13->m128i_i64[0], v8);
        v13[1].m128i_i64[0] = (__int64)v28;
        sub_73D8E0(a1, 0x5Bu, *v28, 0, (__int64)v13);
        return sub_724E30((__int64)&v30);
      }
      v29 = v27;
      v16 = sub_7E8090(v13, 1u);
      *(_QWORD *)(v8 + 16) = v29;
    }
    v19 = (__int64 *)sub_73DBF0(0x49u, v13->m128i_i64[0], v8);
    v20 = sub_73DF90((__int64)v13, v19);
    *((_QWORD *)v20 + 2) = v16;
    sub_73D8E0(a1, 0x5Bu, *v16, 0, (__int64)v20);
  }
  return sub_724E30((__int64)&v30);
}
