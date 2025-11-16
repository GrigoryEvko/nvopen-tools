// Function: sub_2162600
// Address: 0x2162600
//
__int64 __fastcall sub_2162600(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 *v7; // rax
  __int64 *v11; // r14
  __int64 v12; // rsi
  __int64 v13; // r15
  char v14; // dl
  char v15; // al
  int v16; // eax
  bool v17; // zf
  char v18; // dl
  bool v19; // r8
  __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v29; // r13
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v36; // [rsp+8h] [rbp-88h]
  char v37; // [rsp+8h] [rbp-88h]
  char v38; // [rsp+12h] [rbp-7Eh]
  bool v39; // [rsp+13h] [rbp-7Dh]
  bool v40; // [rsp+14h] [rbp-7Ch]
  bool v41; // [rsp+15h] [rbp-7Bh]
  bool v42; // [rsp+16h] [rbp-7Ah]
  char v43; // [rsp+17h] [rbp-79h]
  char v44; // [rsp+18h] [rbp-78h]
  __int64 *v46; // [rsp+28h] [rbp-68h]
  __m128i v47; // [rsp+30h] [rbp-60h] BYREF
  __int64 v48; // [rsp+40h] [rbp-50h]
  __int64 v49; // [rsp+48h] [rbp-48h]
  __int64 v50; // [rsp+50h] [rbp-40h]

  v7 = a2 + 3;
  v11 = a2 + 2;
  v12 = *(_QWORD *)(a1 + 8);
  v46 = v7;
  v13 = a2[7];
  if ( a6 )
  {
    v14 = *(_BYTE *)(a5 + 3);
    v15 = v14 & 0x20;
    if ( (v14 & 0x10) != 0 )
    {
      v16 = v15 == 0 ? 2 : 6;
    }
    else
    {
      LOBYTE(v16) = 4 * (v15 != 0);
      if ( (v14 & 0x40) != 0 )
        LOBYTE(v16) = v16 | 8;
    }
    v44 = 0;
    v17 = (v14 & 0x40) == 0 || (v14 & 0x10) == 0;
    v18 = *(_BYTE *)(a5 + 4);
    if ( !v17 )
      LOBYTE(v16) = v16 | 0x10;
    if ( (v18 & 1) != 0 )
      LOBYTE(v16) = v16 | 0x20;
    if ( (v18 & 4) != 0 )
      LOBYTE(v16) = v16 | 0x40;
    if ( (v18 & 8) != 0 )
      LOBYTE(v16) = v16 | 0x80;
    if ( (*(_BYTE *)(a5 + 4) & 2) != 0 )
      v44 = 1;
    v43 = (unsigned __int8)v16 >> 7;
    v39 = (v16 & 0x40) != 0;
    v42 = (v16 & 0x20) != 0;
    v41 = (v16 & 4) != 0;
    v40 = (v16 & 2) != 0;
    v19 = (v16 & 0x18) != 0;
    if ( a4 )
      goto LABEL_16;
    v37 = (v16 & 0x18) != 0;
    v29 = (__int64)sub_1E0B640(v13, v12 + 12160, a7, 0);
    sub_1DD5BA0(v11, v29);
    v30 = a2[3];
    *(_QWORD *)(v29 + 8) = v46;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v29 = v30 | *(_QWORD *)v29 & 7LL;
    *(_QWORD *)(v30 + 8) = v29;
    v31 = a2[3];
    v47.m128i_i8[0] = 0;
    v48 = 0;
    v49 = 0;
    a2[3] = v29 | v31 & 7;
    LODWORD(v30) = *(_DWORD *)(a5 + 8);
    v50 = 0;
    v47.m128i_i32[2] = v30;
    v47.m128i_i8[3] = (v37 << 6) | (32 * v41) & 0x3F | (16 * v40) & 0x3F | v47.m128i_i8[3] & 0xF;
    v47.m128i_i16[1] &= 0xF00Fu;
    v47.m128i_i32[0] &= 0xFFF000FF;
    v47.m128i_i8[4] = (8 * v43) | (4 * v39) | (2 * v44) & 0xF3 | v42 & 0xF3 | v47.m128i_i8[4] & 0xF0;
    sub_1E1A9C0(v29, v13, &v47);
    v47.m128i_i8[0] = 4;
    v49 = a3;
    v47.m128i_i32[0] &= 0xFFF000FF;
    v48 = 0;
    sub_1E1A9C0(v29, v13, &v47);
    return 1;
  }
  else
  {
    if ( a4 )
    {
      v40 = 0;
      v19 = 0;
      v41 = 0;
      v42 = 0;
      v39 = 0;
      v43 = 0;
      v44 = 0;
LABEL_16:
      v38 = v19;
      v36 = (__int64)sub_1E0B640(v13, v12 + 12160, a7, 0);
      sub_1DD5BA0(v11, v36);
      v20 = *(_QWORD *)v36;
      v21 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v36 + 8) = v46;
      *(_QWORD *)v36 = v21 | v20 & 7;
      *(_QWORD *)(v21 + 8) = v36;
      v22 = a2[3];
      v47.m128i_i8[0] = 0;
      v48 = 0;
      v23 = v22 & 7;
      v49 = 0;
      v50 = 0;
      LOBYTE(v22) = v47.m128i_i8[3] & 0xCF;
      a2[3] = v36 | v23;
      v47.m128i_i32[2] = *(_DWORD *)(a5 + 8);
      v47.m128i_i8[3] = (v38 << 6) | ((32 * v41) | (16 * v40) | v22) & 0x3F;
      v47.m128i_i16[1] &= 0xF00Fu;
      v47.m128i_i32[0] &= 0xFFF000FF;
      v47.m128i_i8[4] = (8 * v43) | (4 * v39) | (2 * v44) & 0xF3 | v42 & 0xF3 | v47.m128i_i8[4] & 0xF0;
      sub_1E1A9C0(v36, v13, &v47);
      v47.m128i_i8[0] = 4;
      v49 = a3;
      v47.m128i_i32[0] &= 0xFFF000FF;
      v48 = 0;
      sub_1E1A9C0(v36, v13, &v47);
      v24 = a2[7];
      v25 = (__int64)sub_1E0B640(v24, *(_QWORD *)(a1 + 8) + 34112LL, a7, 0);
      sub_1DD5BA0(v11, v25);
      v26 = a2[3];
      *(_QWORD *)(v25 + 8) = v46;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v25 = v26 | *(_QWORD *)v25 & 7LL;
      *(_QWORD *)(v26 + 8) = v25;
      v27 = a2[3];
      v47.m128i_i8[0] = 4;
      v47.m128i_i32[0] &= 0xFFF000FF;
      v48 = 0;
      a2[3] = v25 | v27 & 7;
      v49 = a4;
      sub_1E1A9C0(v25, v24, &v47);
      return 2;
    }
    v32 = (__int64)sub_1E0B640(a2[7], v12 + 34112, a7, 0);
    sub_1DD5BA0(v11, v32);
    v33 = a2[3];
    *(_QWORD *)(v32 + 8) = v46;
    v33 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v32 = v33 | *(_QWORD *)v32 & 7LL;
    *(_QWORD *)(v33 + 8) = v32;
    v34 = a2[3];
    v47.m128i_i8[0] = 4;
    v47.m128i_i32[0] &= 0xFFF000FF;
    v48 = 0;
    a2[3] = v32 | v34 & 7;
    v49 = a3;
    sub_1E1A9C0(v32, v13, &v47);
    return 1;
  }
}
