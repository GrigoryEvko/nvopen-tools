// Function: sub_3224ED0
// Address: 0x3224ed0
//
__int64 __fastcall sub_3224ED0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        __int64 (__fastcall *a6)(_QWORD))
{
  void (*v10)(); // r13
  __int128 v11; // rax
  __int64 v12; // r8
  _QWORD *v13; // rdi
  __int64 v14; // r8
  void (*v15)(); // rax
  _QWORD *v16; // r15
  __int64 v17; // rdi
  void (*v18)(); // rax
  unsigned __int8 v19; // al
  __int64 *v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdi
  __m128i v24; // rax
  unsigned __int8 v25; // al
  __int64 v26; // rcx
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  __m128i v35; // xmm1
  char v36; // al
  __int128 v37; // xmm0
  char v38; // dl
  __int64 v39; // rsi
  unsigned __int8 v40; // al
  __int64 v41; // rdx
  __int64 v42; // r14
  void (*v43)(); // r12
  __int128 v44; // rax
  __int64 v46; // [rsp+30h] [rbp-130h]
  unsigned __int16 v47; // [rsp+44h] [rbp-11Ch]
  __int64 v48; // [rsp+48h] [rbp-118h]
  __int64 v50; // [rsp+58h] [rbp-108h]
  __int64 v51; // [rsp+60h] [rbp-100h]
  void (__fastcall *v52)(_QWORD *, __int64, _QWORD, _QWORD); // [rsp+60h] [rbp-100h]
  char v53; // [rsp+6Bh] [rbp-F5h]
  _QWORD v55[2]; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v56[2]; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v57; // [rsp+90h] [rbp-D0h] BYREF
  char v58; // [rsp+98h] [rbp-C8h]
  __m128i v59; // [rsp+A0h] [rbp-C0h] BYREF
  char v60; // [rsp+B0h] [rbp-B0h]
  __int128 v61; // [rsp+C0h] [rbp-A0h]
  char v62; // [rsp+D0h] [rbp-90h]
  __m128i v63; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v64; // [rsp+F0h] [rbp-70h]
  __int128 v65; // [rsp+100h] [rbp-60h] BYREF
  __int64 v66; // [rsp+110h] [rbp-50h]
  __int16 v67; // [rsp+120h] [rbp-40h]

  v51 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v10 = *(void (**)())(*(_QWORD *)v51 + 120LL);
  *(_QWORD *)&v11 = a6(a4);
  v12 = v51;
  v67 = 261;
  v65 = v11;
  if ( v10 != nullsub_98 )
    ((void (__fastcall *)(__int64, __int128 *, __int64))v10)(v51, &v65, 1);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 424LL))(
    *(_QWORD *)(a1 + 8),
    a4,
    0,
    0,
    v12);
  v13 = *(_QWORD **)(a1 + 8);
  v14 = v13[28];
  v15 = *(void (**)())(*(_QWORD *)v14 + 120LL);
  *(_QWORD *)&v65 = "Line Number";
  v67 = 259;
  if ( v15 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __int128 *, __int64))v15)(v14, &v65, 1);
    v13 = *(_QWORD **)(a1 + 8);
  }
  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*v13 + 424LL))(v13, *(unsigned int *)(a2 + 4), 0, 0);
  v16 = *(_QWORD **)(a1 + 8);
  v17 = v16[28];
  v18 = *(void (**)())(*(_QWORD *)v17 + 120LL);
  *(_QWORD *)&v65 = "File Number";
  v67 = 259;
  if ( v18 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __int128 *, __int64))v18)(v17, &v65, 1);
    v16 = *(_QWORD **)(a1 + 8);
  }
  v50 = a2 - 16;
  v19 = *(_BYTE *)(a2 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(__int64 **)(a2 - 32);
  else
    v20 = (__int64 *)(v50 - 8LL * ((v19 >> 2) & 0xF));
  v21 = *v20;
  v52 = *(void (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))(*v16 + 424LL);
  if ( !*(_BYTE *)(a1 + 3769) )
  {
    v39 = (unsigned int)sub_373B2C0(a3, v21);
    goto LABEL_21;
  }
  v22 = sub_3222C60(a1, a3);
  v23 = *(_QWORD *)(v21 + 40);
  v53 = 0;
  v48 = v22;
  if ( v23 )
  {
    v24.m128i_i64[0] = sub_B91420(v23);
    v53 = 1;
    v63 = v24;
  }
  v47 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL) + 1904LL);
  sub_3222AF0(&v59, a1, v21);
  v25 = *(_BYTE *)(v21 - 16);
  v26 = v21 - 16;
  if ( (v25 & 2) != 0 )
  {
    v27 = **(_QWORD **)(v21 - 32);
    if ( !v27 )
    {
      v30 = 0;
      goto LABEL_15;
    }
  }
  else
  {
    v27 = *(_QWORD *)(v26 - 8LL * ((v25 >> 2) & 0xF));
    if ( !v27 )
    {
      v30 = 0;
      goto LABEL_31;
    }
  }
  v28 = sub_B91420(v27);
  v26 = v21 - 16;
  v27 = v28;
  v25 = *(_BYTE *)(v21 - 16);
  v30 = v29;
  if ( (v25 & 2) == 0 )
  {
LABEL_31:
    v31 = v26 - 8LL * ((v25 >> 2) & 0xF);
    goto LABEL_16;
  }
LABEL_15:
  v31 = *(_QWORD *)(v21 - 32);
LABEL_16:
  v32 = *(_QWORD *)(v31 + 8);
  if ( v32 )
  {
    v46 = v27;
    v33 = sub_B91420(v32);
    v27 = v46;
    v32 = v33;
  }
  else
  {
    v34 = 0;
  }
  v55[1] = v34;
  v56[1] = v30;
  v35 = _mm_loadu_si128(&v63);
  LOBYTE(v64) = v53;
  v65 = (__int128)v35;
  v36 = v60;
  v37 = (__int128)_mm_loadu_si128(&v59);
  v55[0] = v32;
  v66 = v64;
  *(_BYTE *)(v48 + 520) = 1;
  v56[0] = v27;
  v62 = v36;
  v61 = v37;
  sub_E78AD0((__int64)&v57, v48, (__int64)v55, (__int64)v56, v47, 0, v37, v36, v65, v66);
  v38 = v58 & 1;
  v58 = (2 * (v58 & 1)) | v58 & 0xFD;
  if ( v38 )
    BUG();
  v39 = v57;
LABEL_21:
  v52(v16, v39, 0, 0);
  v40 = *(_BYTE *)(a2 - 16);
  if ( (v40 & 2) != 0 )
    v41 = *(_QWORD *)(a2 - 32);
  else
    v41 = v50 - 8LL * ((v40 >> 2) & 0xF);
  sub_32253E0(a1, *(_QWORD *)(v41 + 8), a3);
  v42 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v43 = *(void (**)())(*(_QWORD *)v42 + 120LL);
  *(_QWORD *)&v44 = a6(a5);
  v67 = 261;
  v65 = v44;
  if ( v43 != nullsub_98 )
    ((void (__fastcall *)(__int64, __int128 *, __int64))v43)(v42, &v65, 1);
  return (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 424LL))(
           *(_QWORD *)(a1 + 8),
           a5,
           0,
           0);
}
