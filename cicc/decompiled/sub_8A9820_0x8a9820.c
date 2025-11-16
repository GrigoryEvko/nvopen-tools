// Function: sub_8A9820
// Address: 0x8a9820
//
void __fastcall sub_8A9820(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  int v8; // eax
  _QWORD *v9; // r13
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __int64 v13; // rax
  char v14; // al
  __int64 *v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 *v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 *v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r15
  int v34; // edi
  __int64 v35; // rax
  __int64 v36; // [rsp+0h] [rbp-4E0h]
  __int64 v37; // [rsp+8h] [rbp-4D8h]
  __int64 v38; // [rsp+18h] [rbp-4C8h] BYREF
  __m128i v39; // [rsp+20h] [rbp-4C0h] BYREF
  char v40; // [rsp+31h] [rbp-4AFh]
  __int64 v41; // [rsp+38h] [rbp-4A8h]
  __int64 *v42[14]; // [rsp+60h] [rbp-480h] BYREF
  _QWORD v43[60]; // [rsp+D0h] [rbp-410h] BYREF
  __m128i v44; // [rsp+2B0h] [rbp-230h] BYREF
  unsigned int v45; // [rsp+2C4h] [rbp-21Ch]
  int v46; // [rsp+2C8h] [rbp-218h]
  unsigned int v47; // [rsp+2E4h] [rbp-1FCh]
  int v48; // [rsp+2F8h] [rbp-1E8h]
  __int64 v49; // [rsp+33Ch] [rbp-1A4h]
  int v50; // [rsp+358h] [rbp-188h]
  __int64 v51; // [rsp+370h] [rbp-170h]
  int v52; // [rsp+37Ch] [rbp-164h]
  __int64 v53; // [rsp+3A0h] [rbp-140h]
  __m128i v54; // [rsp+3D0h] [rbp-110h]
  __m128i v55; // [rsp+3E0h] [rbp-100h]
  _QWORD *v56; // [rsp+400h] [rbp-E0h]
  _QWORD v57[27]; // [rsp+408h] [rbp-D8h] BYREF

  v6 = *(_QWORD *)(a1->m128i_i64[1] + 88);
  v37 = a1->m128i_i64[1];
  v7 = *(_QWORD *)(v6 + 32);
  sub_864700(*(_QWORD *)(v7 + 24), a3, 0, a2, a4, a5, 1, 0);
  sub_88DA60((__int64 ***)v7, 0);
  if ( dword_4F077C4 == 2 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
  memset(v43, 0, 0x1D8u);
  v43[19] = v43;
  v43[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v43[22]) |= 1u;
  v36 = a1[3].m128i_i64[1];
  sub_891F00((__int64)&v44, (__int64)v43);
  v45 = 1;
  v8 = *(_DWORD *)(v36 + 168);
  v53 = a3;
  v51 = v7;
  v50 = v8;
  v52 = 0;
  unk_4F072F5 = 1;
  v9 = sub_727340();
  v9[8] = *(_QWORD *)(v44.m128i_i64[0] + 24);
  *(_QWORD *)((char *)v9 + 140) = v49;
  if ( !dword_4F04C3C )
  {
    sub_8699D0((__int64)v9, 59, 0);
    dword_4F06C5C = 0;
  }
  v56 = v9;
  v10 = *(_QWORD *)(v7 + 32);
  v48 = 1;
  v9[22] = v10;
  v11 = _mm_loadu_si128(a1 + 1);
  v57[18] = v10;
  v12 = _mm_loadu_si128(a1 + 2);
  v13 = a1->m128i_i64[1];
  v54 = v11;
  v55 = v12;
  v57[15] = v13;
  sub_854C10(*(const __m128i **)(v6 + 56));
  sub_7BC160((__int64)a1[1].m128i_i64);
  v14 = *(_BYTE *)(v37 + 80);
  if ( v14 == 19 || (unsigned __int8)(v14 - 4) <= 1u )
  {
    sub_8A6CC0(&v44, (unsigned __int64 *)v42, &v39, 1);
    if ( dword_4F077C4 != 2 )
      goto LABEL_11;
    goto LABEL_19;
  }
  v33 = v44.m128i_i64[0];
  sub_87E3B0((__int64)v42);
  sub_898140(v33, 1u, v45, 0, v53, v47, v46, &v39, (unsigned __int64)v42, 0, 0, v57);
  if ( v47 )
  {
    v40 |= 0x20u;
    v41 = 0;
  }
  v42[0] = (__int64 *)sub_89D8A0((__int64)&v44, (__int64)&v39, &v38);
  if ( dword_4F077C4 == 2 )
  {
LABEL_19:
    v34 = dword_4F04C40;
    v35 = 776LL * (int)dword_4F04C40;
    *(_BYTE *)(qword_4F04C68[0] + v35 + 7) &= ~8u;
    if ( *(_QWORD *)(qword_4F04C68[0] + v35 + 456) )
      sub_8845B0(v34);
  }
LABEL_11:
  v15 = v42[0];
  v16 = v42[0][11];
  sub_854980((__int64)v42[0], 0);
  while ( word_4F06418[0] != 9 )
    sub_7B8B50((unsigned __int64)v15, 0, v17, v18, v19, v20);
  sub_7B8B50((unsigned __int64)v15, 0, v17, v18, v19, v20);
  sub_863FC0((__int64)v15, 0, v21, v22, v23, v24);
  sub_863FE0((__int64)v15, 0, v25, v26, v27, v28);
  sub_897580((__int64)&v44, v42[0], v16);
  if ( (*(_BYTE *)(*(_QWORD *)(v6 + 104) + 121LL) & 1) != 0 )
    *(_BYTE *)(*(_QWORD *)(v16 + 104) + 121LL) |= 1u;
  else
    sub_88F380((__int64)&v44, v6);
  sub_8911B0((__int64)&v44, (__int64)v42[0], v29, v30, v31, v32);
}
