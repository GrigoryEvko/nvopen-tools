// Function: sub_C24C60
// Address: 0xc24c60
//
__int64 __fastcall sub_C24C60(__int64 a1, __int64 *a2, __int64 a3, int a4, int a5, __int64 a6, __int64 a7, __int64 a8)
{
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r15
  __int64 v15; // rax
  __m128i *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rcx
  __m128i *v19; // rdi
  __int64 *v20; // r13
  __int64 v21; // r14
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // r14
  __int64 *v33; // r13
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 (__fastcall ***v39)(); // rax
  __int64 v40; // [rsp+0h] [rbp-F0h]
  __int64 v41; // [rsp+0h] [rbp-F0h]
  __int64 v42; // [rsp+8h] [rbp-E8h]
  __int64 v43; // [rsp+8h] [rbp-E8h]
  __int64 v44; // [rsp+8h] [rbp-E8h]
  int v46; // [rsp+10h] [rbp-E0h]
  __int64 v47; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v48; // [rsp+28h] [rbp-C8h]
  char v49; // [rsp+30h] [rbp-C0h]
  _QWORD v50[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v51; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v52[4]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v53; // [rsp+80h] [rbp-70h]
  __int64 v54[2]; // [rsp+90h] [rbp-60h] BYREF
  _QWORD v55[2]; // [rsp+A0h] [rbp-50h] BYREF
  int v56; // [rsp+B0h] [rbp-40h]
  __int64 *v57; // [rsp+B8h] [rbp-38h]

  if ( !sub_C216F0(*a2) )
  {
    if ( !sub_C21770(*a2) )
    {
      if ( (unsigned __int8)sub_C21B20(*a2) )
      {
        v35 = *a2;
        *a2 = 0;
        v52[0] = v35;
        v36 = sub_22077B0(288);
        v14 = v36;
        if ( v36 )
        {
          *(_QWORD *)(v36 + 64) = a3;
          v37 = v52[0];
          *(_OWORD *)(v14 + 40) = 0;
          *(_OWORD *)(v14 + 24) = 0;
          *(_QWORD *)(v14 + 72) = v37;
          *(_QWORD *)v14 = &unk_49DBA88;
          *(_QWORD *)(v14 + 104) = v14 + 152;
          *(_QWORD *)(v14 + 8) = v14 + 56;
          *(_QWORD *)(v14 + 196) = 0x1F00000000LL;
          *(_WORD *)(v14 + 204) = 0;
          *(_DWORD *)(v14 + 40) = 1065353216;
          *(_DWORD *)(v14 + 136) = 1065353216;
          v52[0] = 0;
          *(_QWORD *)(v14 + 56) = 0;
          *(_QWORD *)(v14 + 16) = 1;
          *(_QWORD *)(v14 + 48) = 0;
          v54[0] = 0;
          *(_QWORD *)(v14 + 80) = 0;
          *(_QWORD *)(v14 + 88) = 0;
          *(_QWORD *)(v14 + 96) = 0;
          *(_QWORD *)(v14 + 112) = 1;
          *(_QWORD *)(v14 + 120) = 0;
          *(_QWORD *)(v14 + 128) = 0;
          *(_QWORD *)(v14 + 144) = 0;
          *(_QWORD *)(v14 + 152) = 0;
          *(_QWORD *)(v14 + 160) = 0;
          *(_QWORD *)(v14 + 168) = 0;
          *(_QWORD *)(v14 + 176) = 0;
          *(_BYTE *)(v14 + 184) = 0;
          *(_QWORD *)(v14 + 188) = 3;
          sub_C21E00(v54);
          *(_QWORD *)(v14 + 208) = 0;
          *(_WORD *)(v14 + 224) = 0;
          *(_QWORD *)(v14 + 216) = 0;
          *(_QWORD *)v14 = &unk_49DBD38;
          v38 = *(_QWORD *)(v14 + 72);
          *(_QWORD *)(v14 + 232) = 0;
          *(_QWORD *)(v14 + 240) = 1;
          v44 = v38;
          v54[0] = 0;
          sub_9C66B0(v54);
          *(_DWORD *)(v14 + 256) = 0;
          *(_QWORD *)(v14 + 264) = 0;
          *(_QWORD *)(v14 + 248) = v44;
          *(_QWORD *)(v14 + 272) = 0;
          *(_QWORD *)(v14 + 280) = 0;
        }
      }
      else
      {
        if ( !(unsigned __int8)sub_C20A90(*a2) )
        {
          *(_BYTE *)(a1 + 16) |= 1u;
          v39 = sub_C1AFD0();
          *(_DWORD *)a1 = 6;
          *(_QWORD *)(a1 + 8) = v39;
          return a1;
        }
        v28 = *a2;
        *a2 = 0;
        v52[0] = v28;
        v14 = sub_22077B0(232);
        if ( v14 )
        {
          v29 = v52[0];
          *(_QWORD *)(v14 + 56) = 0;
          *(_OWORD *)(v14 + 40) = 0;
          *(_QWORD *)(v14 + 72) = v29;
          *(_QWORD *)v14 = &unk_49DBA88;
          *(_QWORD *)(v14 + 8) = v14 + 56;
          *(_QWORD *)(v14 + 104) = v14 + 152;
          *(_QWORD *)(v14 + 196) = 0x1F00000000LL;
          *(_OWORD *)(v14 + 24) = 0;
          *(_QWORD *)(v14 + 16) = 1;
          *(_QWORD *)(v14 + 48) = 0;
          *(_QWORD *)(v14 + 64) = a3;
          *(_QWORD *)(v14 + 80) = 0;
          *(_QWORD *)(v14 + 88) = 0;
          *(_QWORD *)(v14 + 96) = 0;
          *(_QWORD *)(v14 + 112) = 1;
          *(_QWORD *)(v14 + 120) = 0;
          *(_QWORD *)(v14 + 128) = 0;
          *(_QWORD *)(v14 + 144) = 0;
          *(_QWORD *)(v14 + 152) = 0;
          *(_QWORD *)(v14 + 160) = 0;
          *(_QWORD *)(v14 + 168) = 0;
          *(_QWORD *)(v14 + 176) = 0;
          *(_BYTE *)(v14 + 184) = 0;
          *(_QWORD *)(v14 + 188) = 1;
          *(_WORD *)(v14 + 204) = 0;
          *(_DWORD *)(v14 + 40) = 1065353216;
          *(_DWORD *)(v14 + 136) = 1065353216;
          v52[0] = 0;
          v54[0] = 0;
          sub_C21E00(v54);
          *(_QWORD *)(v14 + 224) = 0;
          *(_QWORD *)v14 = &unk_49DBAF0;
          *(_QWORD *)(v14 + 216) = v14 + 208;
          *(_QWORD *)(v14 + 208) = v14 + 208;
        }
      }
      sub_C21E00(v52);
LABEL_21:
      if ( a8 )
        goto LABEL_4;
LABEL_22:
      v22 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 16LL))(v14);
      if ( !v22 )
      {
        *(_BYTE *)(a1 + 16) &= ~1u;
        *(_QWORD *)a1 = v14;
        *(_DWORD *)(v14 + 200) = 6 * a5 + 7;
        return a1;
      }
      *(_BYTE *)(a1 + 16) |= 1u;
      *(_DWORD *)a1 = v22;
      *(_QWORD *)(a1 + 8) = v23;
      goto LABEL_24;
    }
    v24 = *a2;
    *a2 = 0;
    v43 = v24;
    v25 = sub_22077B0(520);
    v13 = v43;
    v14 = v25;
    if ( v25 )
    {
      *(_QWORD *)(v25 + 56) = 0;
      v26 = v25 + 56;
      *(_OWORD *)(v26 - 32) = 0;
      *(_OWORD *)(v26 - 16) = 0;
      *(_QWORD *)(v14 + 8) = v26;
      *(_QWORD *)(v14 + 104) = v14 + 152;
      *(_QWORD *)(v14 + 196) = 0x1F00000000LL;
      *(_QWORD *)(v14 + 16) = 1;
      *(_QWORD *)(v14 + 48) = 0;
      *(_QWORD *)(v14 + 64) = a3;
      *(_QWORD *)(v14 + 72) = v43;
      *(_QWORD *)(v14 + 80) = 0;
      *(_QWORD *)(v14 + 88) = 0;
      *(_QWORD *)(v14 + 96) = 0;
      *(_QWORD *)(v14 + 112) = 1;
      *(_QWORD *)(v14 + 120) = 0;
      *(_QWORD *)(v14 + 128) = 0;
      *(_QWORD *)(v14 + 144) = 0;
      *(_QWORD *)(v14 + 152) = 0;
      *(_QWORD *)(v14 + 160) = 0;
      *(_QWORD *)(v14 + 168) = 0;
      *(_QWORD *)(v14 + 176) = 0;
      *(_BYTE *)(v14 + 184) = 0;
      *(_QWORD *)(v14 + 188) = 4;
      *(_WORD *)(v14 + 204) = 0;
      *(_QWORD *)(v14 + 208) = 0;
      *(_QWORD *)(v14 + 216) = 0;
      *(_QWORD *)(v14 + 224) = 0;
      *(_QWORD *)(v14 + 232) = 0;
      *(_QWORD *)(v14 + 240) = 0;
      *(_DWORD *)(v14 + 40) = 1065353216;
      *(_DWORD *)(v14 + 136) = 1065353216;
      *(_QWORD *)(v14 + 248) = 0;
      *(_QWORD *)(v14 + 320) = v14 + 336;
      *(_QWORD *)(v14 + 328) = 0x400000000LL;
      *(_QWORD *)(v14 + 368) = v14 + 384;
      *(_QWORD *)(v14 + 256) = 0;
      *(_QWORD *)(v14 + 264) = 0;
      *(_QWORD *)(v14 + 272) = 0;
      *(_QWORD *)(v14 + 280) = 0;
      *(_QWORD *)(v14 + 288) = 0;
      *(_QWORD *)(v14 + 296) = 0;
      *(_QWORD *)(v14 + 304) = 0;
      *(_QWORD *)(v14 + 312) = 0;
      *(_QWORD *)(v14 + 376) = 0;
      *(_QWORD *)(v14 + 384) = 0;
      *(_QWORD *)(v14 + 392) = 1;
      *(_QWORD *)(v14 + 400) = 0;
      *(_QWORD *)(v14 + 408) = 0;
      *(_QWORD *)(v14 + 416) = 0;
      *(_QWORD *)(v14 + 424) = 0;
      *(_QWORD *)(v14 + 432) = 0;
      *(_QWORD *)(v14 + 440) = 0;
      *(_QWORD *)(v14 + 448) = 0;
      *(_DWORD *)(v14 + 456) = 0;
      *(_QWORD *)(v14 + 464) = 0;
      *(_QWORD *)(v14 + 472) = 0;
      *(_QWORD *)(v14 + 480) = 0;
      *(_QWORD *)(v14 + 488) = 0;
      *(_QWORD *)(v14 + 496) = 0;
      *(_QWORD *)(v14 + 504) = 0;
      *(_DWORD *)(v14 + 512) = 0;
      *(_QWORD *)v14 = &unk_49DBBC8;
      goto LABEL_21;
    }
LABEL_19:
    if ( v13 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
    goto LABEL_21;
  }
  v11 = *a2;
  *a2 = 0;
  v42 = v11;
  v12 = sub_22077B0(304);
  v13 = v42;
  v14 = v12;
  if ( !v12 )
    goto LABEL_19;
  *(_QWORD *)(v12 + 56) = 0;
  v15 = v12 + 56;
  *(_OWORD *)(v15 - 32) = 0;
  *(_OWORD *)(v15 - 16) = 0;
  *(_QWORD *)(v14 + 8) = v15;
  *(_QWORD *)(v14 + 72) = v42;
  *(_QWORD *)(v14 + 104) = v14 + 152;
  *(_QWORD *)(v14 + 196) = 0x1F00000000LL;
  *(_QWORD *)(v14 + 16) = 1;
  *(_QWORD *)(v14 + 48) = 0;
  *(_QWORD *)(v14 + 64) = a3;
  *(_QWORD *)(v14 + 80) = 0;
  *(_QWORD *)(v14 + 88) = 0;
  *(_QWORD *)(v14 + 96) = 0;
  *(_QWORD *)(v14 + 112) = 1;
  *(_QWORD *)(v14 + 120) = 0;
  *(_QWORD *)(v14 + 128) = 0;
  *(_QWORD *)(v14 + 144) = 0;
  *(_QWORD *)(v14 + 152) = 0;
  *(_QWORD *)(v14 + 160) = 0;
  *(_QWORD *)(v14 + 168) = 0;
  *(_QWORD *)(v14 + 176) = 0;
  *(_BYTE *)(v14 + 184) = 0;
  *(_QWORD *)(v14 + 188) = 255;
  *(_WORD *)(v14 + 204) = 0;
  *(_QWORD *)(v14 + 208) = 0;
  *(_QWORD *)(v14 + 216) = 0;
  *(_QWORD *)(v14 + 224) = 0;
  *(_QWORD *)(v14 + 232) = 0;
  *(_QWORD *)(v14 + 240) = 0;
  *(_DWORD *)(v14 + 40) = 1065353216;
  *(_DWORD *)(v14 + 136) = 1065353216;
  *(_QWORD *)(v14 + 248) = 0;
  *(_QWORD *)(v14 + 256) = 0;
  *(_QWORD *)(v14 + 264) = 0;
  *(_QWORD *)(v14 + 272) = 0;
  *(_QWORD *)(v14 + 280) = 0;
  *(_QWORD *)(v14 + 288) = 0;
  *(_QWORD *)(v14 + 296) = 0;
  *(_QWORD *)v14 = &unk_49DBB58;
  if ( !a8 )
    goto LABEL_22;
LABEL_4:
  sub_C24BC0((__int64)&v47, a7, a8, a4, v14, a3);
  if ( (v49 & 1) == 0 || (v46 = v47) == 0 )
  {
    v30 = v47;
    v31 = *(_QWORD *)(v14 + 88);
    v47 = 0;
    *(_QWORD *)(v14 + 88) = v30;
    if ( v31 )
    {
      sub_C7D6A0(*(_QWORD *)(v31 + 24), 24LL * *(unsigned int *)(v31 + 40), 8);
      v32 = *(_QWORD *)(v31 + 8);
      if ( v32 )
      {
        sub_EE5E50(*(_QWORD *)(v31 + 8));
        j_j___libc_free_0(v32, 8);
      }
      sub_C21E00((__int64 *)v31);
      j_j___libc_free_0(v31, 64);
      if ( (v49 & 1) == 0 )
      {
        v33 = (__int64 *)v47;
        if ( v47 )
        {
          sub_C7D6A0(*(_QWORD *)(v47 + 24), 24LL * *(unsigned int *)(v47 + 40), 8);
          v34 = v33[1];
          if ( v34 )
          {
            sub_EE5E50(v33[1]);
            j_j___libc_free_0(v34, 8);
          }
          sub_C21E00(v33);
          j_j___libc_free_0(v33, 64);
        }
      }
    }
    goto LABEL_22;
  }
  v40 = v48;
  (*(void (__fastcall **)(__int64 *, __int64))(*(_QWORD *)v48 + 32LL))(v54, v48);
  v16 = (__m128i *)sub_2241130(v54, 0, 0, "Could not create remapper: ", 27);
  v17 = v40;
  v50[0] = &v51;
  if ( (__m128i *)v16->m128i_i64[0] == &v16[1] )
  {
    v51 = _mm_loadu_si128(v16 + 1);
  }
  else
  {
    v50[0] = v16->m128i_i64[0];
    v51.m128i_i64[0] = v16[1].m128i_i64[0];
  }
  v18 = v16->m128i_i64[1];
  v16[1].m128i_i8[0] = 0;
  v50[1] = v18;
  v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
  v16->m128i_i64[1] = 0;
  if ( (_QWORD *)v54[0] != v55 )
  {
    j_j___libc_free_0(v54[0], v55[0] + 1LL);
    v17 = v40;
  }
  v55[1] = a8;
  v53 = 260;
  v52[0] = (__int64)v50;
  v41 = v17;
  v54[1] = 12;
  v54[0] = (__int64)&unk_49D9C78;
  v56 = 0;
  v55[0] = a7;
  v57 = v52;
  sub_B6EB20(a3, (__int64)v54);
  *(_BYTE *)(a1 + 16) |= 1u;
  v19 = (__m128i *)v50[0];
  *(_DWORD *)a1 = v46;
  *(_QWORD *)(a1 + 8) = v41;
  if ( v19 != &v51 )
    j_j___libc_free_0(v19, v51.m128i_i64[0] + 1);
  if ( (v49 & 1) == 0 )
  {
    v20 = (__int64 *)v47;
    if ( v47 )
    {
      sub_C7D6A0(*(_QWORD *)(v47 + 24), 24LL * *(unsigned int *)(v47 + 40), 8);
      v21 = v20[1];
      if ( v21 )
      {
        sub_EE5E50(v20[1]);
        j_j___libc_free_0(v21, 8);
      }
      sub_C21E00(v20);
      j_j___libc_free_0(v20, 64);
    }
  }
  if ( v14 )
LABEL_24:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  return a1;
}
