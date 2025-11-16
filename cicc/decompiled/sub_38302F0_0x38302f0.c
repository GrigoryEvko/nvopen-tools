// Function: sub_38302F0
// Address: 0x38302f0
//
__int64 __fastcall sub_38302F0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // r10
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v9; // rsi
  const __m128i *v10; // rax
  __int64 v11; // rcx
  __m128i v12; // xmm0
  __int128 v13; // xmm1
  unsigned __int16 *v14; // rdx
  __int64 v15; // r13
  int v16; // eax
  __int64 v17; // rdx
  unsigned __int16 *v18; // rdx
  __int64 v19; // r8
  unsigned __int64 v20; // rsi
  int v21; // eax
  unsigned __int16 v22; // ax
  int v23; // eax
  __int64 v24; // r9
  unsigned int v25; // edx
  __int64 v26; // r8
  unsigned int v27; // edx
  unsigned int v28; // edx
  __int64 v29; // r12
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int128 v34; // [rsp-30h] [rbp-110h]
  unsigned int v35; // [rsp+8h] [rbp-D8h]
  char v36; // [rsp+8h] [rbp-D8h]
  __int64 v37; // [rsp+10h] [rbp-D0h]
  __int64 *v38; // [rsp+18h] [rbp-C8h]
  __int64 v39; // [rsp+18h] [rbp-C8h]
  char v40; // [rsp+18h] [rbp-C8h]
  __int128 v41; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v42; // [rsp+40h] [rbp-A0h]
  __int64 v43; // [rsp+60h] [rbp-80h]
  unsigned int v44; // [rsp+70h] [rbp-70h] BYREF
  __int64 v45; // [rsp+78h] [rbp-68h]
  __int64 v46; // [rsp+80h] [rbp-60h] BYREF
  int v47; // [rsp+88h] [rbp-58h]
  __int16 v48; // [rsp+90h] [rbp-50h] BYREF
  __int64 v49; // [rsp+98h] [rbp-48h]
  __int64 v50; // [rsp+A0h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(_QWORD *)(a1[1] + 64);
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v8 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v48, *a1, v7, v5, v6);
    LOWORD(v44) = v49;
    v45 = v50;
  }
  else
  {
    v44 = v8(*a1, v7, v5, v6);
    v45 = v33;
  }
  v9 = *(_QWORD *)(a2 + 80);
  v46 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v46, v9, 1);
  v47 = *(_DWORD *)(a2 + 72);
  v10 = *(const __m128i **)(a2 + 40);
  v11 = v10[2].m128i_i64[1];
  v12 = _mm_loadu_si128(v10);
  v13 = (__int128)_mm_loadu_si128(v10 + 5);
  v14 = (unsigned __int16 *)(*(_QWORD *)(v11 + 48) + 16LL * v10[3].m128i_u32[0]);
  v15 = v10[3].m128i_i64[0];
  v16 = *v14;
  v17 = *((_QWORD *)v14 + 1);
  v48 = v16;
  v49 = v17;
  if ( (_WORD)v16 )
  {
    v18 = word_4456340;
    LOBYTE(v11) = (unsigned __int16)(v16 - 176) <= 0x34u;
    v19 = (unsigned int)v11;
    v20 = word_4456340[v16 - 1];
    v21 = (unsigned __int16)v44;
    if ( (_WORD)v44 )
    {
LABEL_7:
      v37 = 0;
      v22 = word_4456580[v21 - 1];
      goto LABEL_8;
    }
  }
  else
  {
    v20 = sub_3007240((__int64)&v48);
    v21 = (unsigned __int16)v44;
    v19 = HIDWORD(v20);
    v11 = HIDWORD(v20);
    if ( (_WORD)v44 )
      goto LABEL_7;
  }
  v36 = v11;
  v40 = v19;
  v22 = sub_3009970((__int64)&v44, v20, (__int64)v18, v11, v19);
  LOBYTE(v11) = v36;
  v37 = v32;
  LOBYTE(v19) = v40;
LABEL_8:
  LODWORD(v43) = v20;
  BYTE4(v43) = v19;
  v38 = *(__int64 **)(a1[1] + 64);
  v35 = v22;
  if ( (_BYTE)v11 )
  {
    LOWORD(v23) = sub_2D43AD0(v22, v20);
    v25 = v35;
    v26 = 0;
    if ( (_WORD)v23 )
      goto LABEL_10;
  }
  else
  {
    LOWORD(v23) = sub_2D43050(v22, v20);
    v25 = v35;
    v26 = 0;
    if ( (_WORD)v23 )
      goto LABEL_10;
  }
  v23 = sub_3009450(v38, v25, v37, v43, 0, v24);
  HIWORD(v2) = HIWORD(v23);
  v26 = v31;
LABEL_10:
  v39 = v26;
  LOWORD(v2) = v23;
  *(_QWORD *)&v41 = sub_37AE0F0((__int64)a1, v12.m128i_u64[0], v12.m128i_i64[1]);
  *((_QWORD *)&v41 + 1) = v27 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v42 = sub_33FAF80(a1[1], 215, (__int64)&v46, v2, v39, 0, v12);
  *((_QWORD *)&v34 + 1) = v28 | v15 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v34 = v42;
  v29 = sub_340F900((_QWORD *)a1[1], 0xA0u, (__int64)&v46, v44, v45, 0xFFFFFFFF00000000LL, v41, v34, v13);
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
  return v29;
}
