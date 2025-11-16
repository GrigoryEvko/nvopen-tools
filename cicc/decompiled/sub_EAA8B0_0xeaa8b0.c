// Function: sub_EAA8B0
// Address: 0xeaa8b0
//
__int64 __fastcall sub_EAA8B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int *a5, unsigned __int64 a6)
{
  __int64 v9; // r14
  int v10; // edx
  __m128i v11; // xmm1
  __int64 (__fastcall *v12)(__int64, __int64, __int64, __int64, __int64, __int64); // rax
  unsigned int v13; // edx
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned int v16; // r13d
  __m128i *v18; // rdx
  __m128i si128; // xmm0
  __int64 v20; // r8
  int v21; // r13d
  __int64 *v22; // rdi
  _QWORD *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // rsi
  int v28; // eax
  __int64 v29; // rcx
  int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // rdi
  unsigned __int64 **v33; // rax
  __int64 *v34; // rdi
  unsigned int v35; // r13d
  __int64 *v36; // rsi
  __m128i v37; // xmm2
  __int64 v38; // rax
  char v39; // dl
  int v40; // r9d
  int v41; // [rsp+10h] [rbp-250h]
  __int64 v42; // [rsp+30h] [rbp-230h]
  __int64 *v43; // [rsp+38h] [rbp-228h]
  __int64 (__fastcall *v45)(__int64, __int64 *, __int64 *, __int64, __int64, __int64); // [rsp+48h] [rbp-218h]
  unsigned __int8 v46; // [rsp+48h] [rbp-218h]
  __int64 (__fastcall *v47)(__int64, __int64, __int64, __int64, __int64, __int64); // [rsp+48h] [rbp-218h]
  _QWORD v48[3]; // [rsp+50h] [rbp-210h] BYREF
  __int64 v49; // [rsp+68h] [rbp-1F8h] BYREF
  _QWORD v50[2]; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 *v51; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 v52; // [rsp+88h] [rbp-1D8h]
  __int64 v53; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 v54; // [rsp+A0h] [rbp-1C0h] BYREF
  _QWORD *v55; // [rsp+A8h] [rbp-1B8h]
  __int16 v56; // [rsp+C0h] [rbp-1A0h]
  __m128i v57; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v58; // [rsp+E0h] [rbp-180h]
  __m128i *v59; // [rsp+E8h] [rbp-178h]
  __m128i *v60; // [rsp+F0h] [rbp-170h]
  __int64 v61; // [rsp+F8h] [rbp-168h]
  __int64 *v62; // [rsp+100h] [rbp-160h]
  __int64 *v63; // [rsp+110h] [rbp-150h] BYREF
  __m128i v64; // [rsp+118h] [rbp-148h]
  __int64 v65; // [rsp+128h] [rbp-138h] BYREF
  unsigned int v66; // [rsp+130h] [rbp-130h]

  v48[0] = a3;
  v48[1] = a4;
  sub_C93130((__int64 *)&v51, (__int64)v48);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *a5;
  v11 = _mm_loadu_si128((const __m128i *)(a5 + 2));
  v49 = *(_QWORD *)(a2 + 88);
  v12 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 56LL);
  LODWORD(v63) = v10;
  v13 = a5[8];
  v64 = v11;
  v66 = v13;
  if ( v13 > 0x40 )
  {
    v47 = v12;
    sub_C43780((__int64)&v65, (const void **)a5 + 3);
    v12 = v47;
  }
  else
  {
    v65 = *((_QWORD *)a5 + 3);
  }
  v14 = v52;
  if ( v12 == sub_EA2450 )
  {
    v43 = v51;
    v45 = *(__int64 (__fastcall **)(__int64, __int64 *, __int64 *, __int64, __int64, __int64))(*(_QWORD *)v9 + 48LL);
    v15 = sub_ECD6A0(&v63);
    v46 = v45(v9, &v49, v43, v14, v15, a2);
  }
  else
  {
    v46 = v12(v9, (__int64)&v49, (__int64)v51, v52, (__int64)&v63, a2);
  }
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  *(_BYTE *)(a2 + 84) = v46;
  if ( *(_BYTE *)(a1 + 33) )
  {
    v63 = &v65;
    v61 = 0x100000000LL;
    v64.m128i_i64[0] = 0;
    v64.m128i_i64[1] = 256;
    v57.m128i_i64[0] = (__int64)&unk_49DD288;
    v57.m128i_i64[1] = 2;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v62 = (__int64 *)&v63;
    sub_CB5980((__int64)&v57, 0, 0, 0);
    v18 = v60;
    if ( (unsigned __int64)((char *)v59 - (char *)v60) <= 0x14 )
    {
      sub_CB6200((__int64)&v57, "parsed instruction: [", 0x15u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F85330);
      v60[1].m128i_i32[0] = 540700271;
      v18[1].m128i_i8[4] = 91;
      *v18 = si128;
      v60 = (__m128i *)((char *)v60 + 21);
    }
    v20 = 0;
    v21 = 0;
    if ( *(_DWORD *)(a2 + 8) )
    {
      while ( 1 )
      {
        (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(*(_QWORD *)a2 + 8 * v20) + 120LL))(
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v20),
          &v57);
        v20 = (unsigned int)(v21 + 1);
        v21 = v20;
        if ( (_DWORD)v20 == *(_DWORD *)(a2 + 8) )
          break;
        if ( (_DWORD)v20 )
        {
          if ( (unsigned __int64)((char *)v59 - (char *)v60) <= 1 )
          {
            v42 = v20;
            sub_CB6200((__int64)&v57, (unsigned __int8 *)", ", 2u);
            v20 = v42;
          }
          else
          {
            v60->m128i_i16[0] = 8236;
            v60 = (__m128i *)((char *)v60 + 2);
          }
        }
      }
    }
    if ( v59 == v60 )
    {
      sub_CB6200((__int64)&v57, (unsigned __int8 *)"]", 1u);
    }
    else
    {
      v60->m128i_i8[0] = 93;
      v60 = (__m128i *)((char *)v60 + 1);
    }
    v22 = *(__int64 **)(a1 + 248);
    v23 = (_QWORD *)v62[1];
    v24 = *v62;
    v56 = 261;
    v55 = v23;
    v54 = v24;
    v50[0] = 0;
    v50[1] = 0;
    sub_C91CB0(v22, a6, 3, (__int64)&v54, (__int64)v50, 1, 0, 0, 1u);
    v57.m128i_i64[0] = (__int64)&unk_49DD388;
    sub_CB5840((__int64)&v57);
    if ( v63 != &v65 )
      _libc_free(v63, a6);
  }
  v16 = v46;
  LOBYTE(v16) = (*(_DWORD *)(a1 + 24) != 0) | v46;
  if ( !(_BYTE)v16 )
  {
    if ( (unsigned __int8)sub_EAA750(a1) )
    {
      v25 = *(_QWORD *)(a1 + 224);
      v26 = *(_QWORD *)(a1 + 232);
      v27 = *(_QWORD *)(v25 + 1808);
      v28 = *(_DWORD *)(v25 + 1824);
      v29 = *(_QWORD *)(*(_QWORD *)(v26 + 288) + 8LL);
      if ( v28 )
      {
        v30 = v28 - 1;
        v31 = (v28 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v32 = *(_QWORD *)(v27 + 8LL * v31);
        if ( v29 == v32 )
        {
LABEL_28:
          v33 = *(unsigned __int64 ***)(a1 + 368);
          v34 = *(__int64 **)(a1 + 248);
          if ( *(unsigned __int64 ***)(a1 + 376) == v33 )
            v35 = sub_C90410(v34, a6, *(_DWORD *)(a1 + 304));
          else
            v35 = sub_C90410(v34, **v33, *((_DWORD *)*v33 + 2));
          if ( *(_QWORD *)(a1 + 488) )
          {
            v36 = *(__int64 **)(a1 + 232);
            v37 = _mm_loadu_si128(&v57);
            v38 = *v36;
            v64.m128i_i8[8] = 0;
            LOBYTE(v58) = 0;
            LOBYTE(v41) = 0;
            (*(void (__fastcall **)(__int64 *, __int64 *, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64, __int64, int, __int64 *, __int64, __int64))(v38 + 656))(
              &v54,
              v36,
              0,
              0,
              0,
              0,
              *(_QWORD *)(a1 + 480),
              *(_QWORD *)(a1 + 488),
              v37.m128i_i64[0],
              v37.m128i_i64[1],
              v41,
              v63,
              v64.m128i_i64[0],
              v64.m128i_i64[1]);
            v39 = (unsigned __int8)v55 & 1;
            LOBYTE(v55) = (2 * ((unsigned __int8)v55 & 1)) | (unsigned __int8)v55 & 0xFD;
            if ( v39 )
              BUG();
            *(_DWORD *)(*(_QWORD *)(a1 + 224) + 1796LL) = v54;
            v35 += *(_DWORD *)(a1 + 496)
                 + ~(unsigned int)sub_C90410(*(__int64 **)(a1 + 248), *(_QWORD *)(a1 + 504), *(_DWORD *)(a1 + 512));
          }
          (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 232) + 688LL))(
            *(_QWORD *)(a1 + 232),
            *(unsigned int *)(*(_QWORD *)(a1 + 224) + 1796LL),
            v35,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0);
          v26 = *(_QWORD *)(a1 + 232);
        }
        else
        {
          v40 = 1;
          while ( v32 != -4096 )
          {
            v31 = v30 & (v40 + v31);
            v32 = *(_QWORD *)(v27 + 8LL * v31);
            if ( v29 == v32 )
              goto LABEL_28;
            ++v40;
          }
        }
      }
    }
    else
    {
      v26 = *(_QWORD *)(a1 + 232);
    }
    v16 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, __int64, __int64 **, _QWORD))(**(_QWORD **)(a1 + 8) + 80LL))(
            *(_QWORD *)(a1 + 8),
            a6,
            a2 + 80,
            a2,
            v26,
            &v63,
            *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 64LL));
  }
  if ( v51 != &v53 )
    j_j___libc_free_0(v51, v53 + 1);
  return v16;
}
