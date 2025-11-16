// Function: sub_258CCE0
// Address: 0x258cce0
//
__int64 __fastcall sub_258CCE0(__int64 a1, __int64 a2)
{
  __int16 v4; // ax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  int v10; // edi
  char v11; // al
  __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned __int8 *v14; // r14
  unsigned __int64 *v15; // rcx
  unsigned __int8 v16; // bl
  char v17; // r12
  __int64 v18; // r13
  __int64 v19; // r15
  __m128i v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 *v23; // r10
  __int64 v24; // rbx
  unsigned __int64 *v25; // r15
  _BYTE *v26; // rdx
  __int64 v27; // rdi
  unsigned __int8 *v28; // r8
  unsigned __int64 v29; // rcx
  __int64 v30; // r14
  __int64 (__fastcall *v31)(__int64); // rax
  __int64 v32; // rax
  unsigned int v33; // r12d
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 *v38; // rbx
  __int64 v39; // [rsp-8h] [rbp-238h]
  _BYTE *v40; // [rsp+0h] [rbp-230h]
  unsigned __int8 *v41; // [rsp+20h] [rbp-210h]
  unsigned __int64 v42; // [rsp+28h] [rbp-208h]
  bool v43; // [rsp+37h] [rbp-1F9h]
  unsigned __int64 v44; // [rsp+38h] [rbp-1F8h]
  __int64 v45; // [rsp+40h] [rbp-1F0h]
  unsigned __int8 *v46; // [rsp+40h] [rbp-1F0h]
  char *v47; // [rsp+48h] [rbp-1E8h]
  __int64 v48; // [rsp+48h] [rbp-1E8h]
  char v49; // [rsp+53h] [rbp-1DDh] BYREF
  int v50; // [rsp+54h] [rbp-1DCh] BYREF
  __int64 v51; // [rsp+58h] [rbp-1D8h] BYREF
  __m128i v52; // [rsp+60h] [rbp-1D0h] BYREF
  __int64 *v53; // [rsp+70h] [rbp-1C0h]
  unsigned __int64 **v54; // [rsp+78h] [rbp-1B8h]
  unsigned __int64 *v55; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v56; // [rsp+88h] [rbp-1A8h]
  _BYTE *v57; // [rsp+90h] [rbp-1A0h]
  char *v58; // [rsp+98h] [rbp-198h]
  __int64 *v59; // [rsp+A0h] [rbp-190h]
  unsigned __int64 v60[2]; // [rsp+B0h] [rbp-180h] BYREF
  _BYTE v61[48]; // [rsp+C0h] [rbp-170h] BYREF
  _QWORD v62[2]; // [rsp+F0h] [rbp-140h] BYREF
  __int16 v63; // [rsp+100h] [rbp-130h]
  __int64 v64; // [rsp+108h] [rbp-128h]
  __int64 v65; // [rsp+110h] [rbp-120h]
  __int64 v66; // [rsp+118h] [rbp-118h]
  __int64 v67; // [rsp+120h] [rbp-110h]
  unsigned __int64 v68[2]; // [rsp+128h] [rbp-108h] BYREF
  _BYTE v69[248]; // [rsp+138h] [rbp-F8h] BYREF

  v40 = (_BYTE *)(a1 + 88);
  v64 = 0;
  v65 = 0;
  v62[0] = &unk_4A171B8;
  v4 = *(_WORD *)(a1 + 104);
  v66 = 0;
  v63 = v4;
  v67 = 0;
  v62[1] = &unk_4A16CD8;
  sub_C7D6A0(0, 0, 8);
  v9 = *(unsigned int *)(a1 + 136);
  LODWORD(v67) = v9;
  if ( (_DWORD)v9 )
  {
    v35 = sub_C7D670(24 * v9, 8);
    v6 = *(_QWORD *)(a1 + 120);
    v65 = v35;
    v5 = v35;
    v66 = *(_QWORD *)(a1 + 128);
    v36 = 0;
    v37 = 24LL * (unsigned int)v67;
    do
    {
      *(__m128i *)(v5 + v36) = _mm_loadu_si128((const __m128i *)(v6 + v36));
      *(_QWORD *)(v5 + v36 + 16) = *(_QWORD *)(v6 + v36 + 16);
      v36 += 24;
    }
    while ( v37 != v36 );
  }
  else
  {
    v65 = 0;
    v66 = 0;
  }
  v10 = *(_DWORD *)(a1 + 152);
  v68[0] = (unsigned __int64)v69;
  v68[1] = 0x800000000LL;
  if ( v10 )
    sub_2539BB0((__int64)v68, a1 + 144, v5, v6, v7, v8);
  v11 = *(_BYTE *)(a1 + 352);
  v49 = 0;
  v69[192] = v11;
  v60[0] = (unsigned __int64)v61;
  v60[1] = 0x300000000LL;
  v12 = sub_25096F0((_QWORD *)(a1 + 72));
  v13 = *(_QWORD *)(a1 + 360);
  v55 = v60;
  v51 = v12;
  v56 = a2;
  v57 = (_BYTE *)a1;
  v58 = &v49;
  v59 = &v51;
  v42 = v13;
  if ( v13 )
  {
    v14 = (unsigned __int8 *)&unk_438A62B;
    v15 = v60;
    v16 = 2;
    while ( 1 )
    {
      *((_DWORD *)v15 + 2) = 0;
      v17 = v16;
      v45 = (__int64)v15;
      v18 = v56;
      v19 = (__int64)v57;
      v47 = v58;
      v20.m128i_i64[0] = sub_250D2C0(v42, 0);
      v52 = v20;
      v22 = v39;
      if ( !(unsigned __int8)sub_2526B50(v18, &v52, v19, v45, v16, v47, 1u) )
        goto LABEL_20;
      if ( v16 == 2 )
      {
        v38 = (__int64 *)(*v55 + 16LL * *((unsigned int *)v55 + 2));
        v22 = (__int64)v38;
        v43 = v38 == sub_2537880((__int64 *)*v55, (__int64)v38, v59);
        v23 = (__int64 *)*v55;
        v24 = *v55 + 16LL * *((unsigned int *)v55 + 2);
        if ( *v55 == v24 )
          goto LABEL_18;
      }
      else
      {
        v23 = (__int64 *)*v55;
        v43 = 0;
        v24 = *v55 + 16LL * *((unsigned int *)v55 + 2);
        if ( *v55 == v24 )
          goto LABEL_19;
      }
      v25 = (unsigned __int64 *)v23;
      v41 = v14;
      if ( v43 )
        v17 = 3;
      do
      {
        v27 = (__int64)v57;
        v28 = (unsigned __int8 *)v25[1];
        v29 = *v25;
        v30 = *v59;
        v31 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v57 + 40LL);
        if ( v31 == sub_2505DE0 )
        {
          v26 = v57 + 88;
        }
        else
        {
          v44 = *v25;
          v46 = (unsigned __int8 *)v25[1];
          v48 = (__int64)v57;
          v32 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64))v31)(v57, v22, v21);
          v29 = v44;
          v28 = v46;
          v27 = v48;
          v26 = (_BYTE *)v32;
        }
        v22 = v56;
        v25 += 2;
        sub_258BA20(v27, v56, v26, v29, v28, v17, v30);
        v21 = v39;
      }
      while ( (unsigned __int64 *)v24 != v25 );
      v14 = v41;
LABEL_18:
      if ( v43 )
        goto LABEL_20;
LABEL_19:
      if ( &unk_438A62D == (_UNKNOWN *)++v14 )
        goto LABEL_20;
      v16 = *v14;
      v15 = v55;
    }
  }
  v53 = &v51;
  v54 = &v55;
  v52.m128i_i64[0] = a2;
  v52.m128i_i64[1] = a1;
  v50 = 1;
  if ( (unsigned __int8)sub_2526370(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_258D9E0,
                          (__int64)&v52,
                          a1,
                          &v50,
                          1,
                          &v49,
                          1,
                          0) )
  {
LABEL_20:
    v33 = (unsigned __int8)sub_255BFA0((__int64)v62, v40);
  }
  else
  {
    v33 = 0;
    *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  }
  if ( (_BYTE *)v60[0] != v61 )
    _libc_free(v60[0]);
  v62[0] = &unk_4A171B8;
  if ( (_BYTE *)v68[0] != v69 )
    _libc_free(v68[0]);
  sub_C7D6A0(v65, 24LL * (unsigned int)v67, 8);
  return v33;
}
