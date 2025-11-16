// Function: sub_383E4F0
// Address: 0x383e4f0
//
__int64 __fastcall sub_383E4F0(__int64 *a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 (*v14)(); // r10
  unsigned __int16 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int16 *v19; // rdx
  int v20; // eax
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  bool v23; // al
  __int64 v24; // r8
  __int16 v25; // ax
  __int64 v26; // rcx
  __int64 v27; // rdx
  int v28; // edx
  unsigned int v30; // edx
  unsigned __int64 v31; // rax
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdi
  unsigned __int16 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  unsigned __int16 *v39; // rdx
  int v40; // eax
  __int64 v41; // rdx
  int v42; // edx
  unsigned int v43; // edx
  unsigned __int64 v44; // rdi
  bool v45; // al
  __int64 v46; // r8
  __int16 v47; // ax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // [rsp+0h] [rbp-120h]
  __int64 v54; // [rsp+8h] [rbp-118h]
  unsigned int v55; // [rsp+8h] [rbp-118h]
  unsigned int v56; // [rsp+10h] [rbp-110h]
  __int64 v57; // [rsp+18h] [rbp-108h]
  unsigned int v58; // [rsp+18h] [rbp-108h]
  __int64 v59; // [rsp+18h] [rbp-108h]
  unsigned int v60; // [rsp+20h] [rbp-100h]
  unsigned int v61; // [rsp+20h] [rbp-100h]
  unsigned __int64 v62; // [rsp+20h] [rbp-100h]
  unsigned int v63; // [rsp+28h] [rbp-F8h]
  __int64 v64; // [rsp+30h] [rbp-F0h]
  __int64 v65; // [rsp+38h] [rbp-E8h]
  __int16 v66; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-78h]
  unsigned __int16 v68; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v69; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v70; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v71; // [rsp+C8h] [rbp-58h]
  unsigned __int64 v72; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v73; // [rsp+D8h] [rbp-48h]
  unsigned __int64 v74; // [rsp+E0h] [rbp-40h]
  unsigned int v75; // [rsp+E8h] [rbp-38h]

  v7 = sub_37AE0F0((__int64)a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8));
  v9 = v8;
  v10 = v7;
  v11 = sub_37AE0F0((__int64)a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8));
  v12 = *a1;
  v65 = v11;
  v64 = v13;
  v14 = *(__int64 (**)())(*(_QWORD *)*a1 + 1456LL);
  if ( v14 == sub_2D56680
    || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v14)(
          v12,
          *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a2 + 48LL) + 16LL * *(unsigned int *)(a2 + 8)),
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 48LL) + 16LL * *(unsigned int *)(a2 + 8) + 8),
          *(unsigned __int16 *)(*(_QWORD *)(v10 + 48) + 16LL * (unsigned int)v9),
          *(_QWORD *)(*(_QWORD *)(v10 + 48) + 16LL * (unsigned int)v9 + 8)) )
  {
    v63 = sub_33DF530(a1[1], v10, v9, 0);
    v60 = sub_33DF530(a1[1], v65, v64, 0);
    v15 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a2 + 48LL) + 16LL * *(unsigned int *)(a2 + 8));
    LODWORD(v16) = *v15;
    v17 = *((_QWORD *)v15 + 1);
    LOWORD(v70) = v16;
    v71 = v17;
    if ( (_WORD)v16 )
    {
      if ( (unsigned __int16)(v16 - 17) <= 0xD3u )
      {
        v17 = 0;
        LOWORD(v16) = word_4456580[(int)v16 - 1];
      }
    }
    else
    {
      v54 = v17;
      v56 = v16;
      v23 = sub_30070B0((__int64)&v70);
      LOWORD(v16) = v56;
      v17 = v54;
      if ( v23 )
      {
        v25 = sub_3009970((__int64)&v70, v65, v56, v54, v24);
        v17 = v16;
        LOWORD(v16) = v25;
      }
    }
    v68 = v16;
    v69 = v17;
    v72 = sub_2D5B750(&v68);
    v73 = v18;
    if ( v63 > v72 )
      goto LABEL_18;
    v19 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * *(unsigned int *)(a3 + 8));
    v20 = *v19;
    v21 = *((_QWORD *)v19 + 1);
    LOWORD(v70) = v20;
    v71 = v21;
    if ( (_WORD)v20 )
    {
      if ( (unsigned __int16)(v20 - 17) > 0xD3u )
      {
        v68 = v20;
        v69 = v21;
        goto LABEL_9;
      }
      LOWORD(v20) = word_4456580[v20 - 1];
      v52 = 0;
    }
    else
    {
      v57 = v21;
      if ( !sub_30070B0((__int64)&v70) )
      {
        v69 = v57;
        v68 = 0;
        goto LABEL_16;
      }
      LOWORD(v20) = sub_3009970((__int64)&v70, v65, v57, v26, (__int64)&v68);
    }
    v68 = v20;
    v69 = v52;
    if ( (_WORD)v20 )
    {
LABEL_9:
      if ( (_WORD)v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
        goto LABEL_70;
      v22 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v20 - 16];
      goto LABEL_17;
    }
LABEL_16:
    v22 = sub_3007260((__int64)&v68);
    v72 = v22;
    v73 = v27;
LABEL_17:
    if ( v60 <= v22 )
      goto LABEL_57;
LABEL_18:
    *(_QWORD *)a2 = sub_37AF270((__int64)a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8), a4);
    *(_DWORD *)(a2 + 8) = v28;
    *(_QWORD *)a3 = sub_37AF270((__int64)a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), a4);
    *(_DWORD *)(a3 + 8) = v30;
    return v30;
  }
  sub_33DD090((__int64)&v72, a1[1], v10, v9, 0);
  v61 = v73;
  if ( (unsigned int)v73 > 0x40 )
  {
    v61 -= sub_C44500((__int64)&v72);
    if ( v75 <= 0x40 || (v44 = v74) == 0 )
    {
LABEL_46:
      if ( v72 )
        j_j___libc_free_0_0(v72);
      goto LABEL_25;
    }
LABEL_45:
    j_j___libc_free_0_0(v44);
    if ( (unsigned int)v73 <= 0x40 )
      goto LABEL_25;
    goto LABEL_46;
  }
  if ( (_DWORD)v73 )
  {
    if ( v72 << (64 - (unsigned __int8)v73) == -1 )
    {
      v61 = v73 - 64;
    }
    else
    {
      _BitScanReverse64(&v31, ~(v72 << (64 - (unsigned __int8)v73)));
      v61 = v73 - (v31 ^ 0x3F);
    }
  }
  if ( v75 > 0x40 )
  {
    v44 = v74;
    if ( v74 )
      goto LABEL_45;
  }
LABEL_25:
  v32 = a1[1];
  sub_33DD090((__int64)&v72, v32, v65, v64, 0);
  v58 = v73;
  if ( (unsigned int)v73 > 0x40 )
  {
    v58 -= sub_C44500((__int64)&v72);
    if ( v75 <= 0x40 || (v34 = v74) == 0 )
    {
LABEL_52:
      if ( v72 )
        j_j___libc_free_0_0(v72);
      goto LABEL_32;
    }
LABEL_31:
    j_j___libc_free_0_0(v34);
    if ( (unsigned int)v73 <= 0x40 )
      goto LABEL_32;
    goto LABEL_52;
  }
  if ( (_DWORD)v73 )
  {
    if ( v72 << (64 - (unsigned __int8)v73) == -1 )
    {
      v58 = v73 - 64;
    }
    else
    {
      _BitScanReverse64(&v33, ~(v72 << (64 - (unsigned __int8)v73)));
      v58 = v73 - (v33 ^ 0x3F);
    }
  }
  if ( v75 > 0x40 )
  {
    v34 = v74;
    if ( v74 )
      goto LABEL_31;
  }
LABEL_32:
  v35 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a2 + 48LL) + 16LL * *(unsigned int *)(a2 + 8));
  LODWORD(v36) = *v35;
  v37 = *((_QWORD *)v35 + 1);
  LOWORD(v70) = v36;
  v71 = v37;
  if ( (_WORD)v36 )
  {
    if ( (unsigned __int16)(v36 - 17) <= 0xD3u )
    {
      v37 = 0;
      LOWORD(v36) = word_4456580[(int)v36 - 1];
    }
  }
  else
  {
    v53 = v37;
    v55 = v36;
    v45 = sub_30070B0((__int64)&v70);
    LOWORD(v36) = v55;
    v37 = v53;
    if ( v45 )
    {
      v47 = sub_3009970((__int64)&v70, v32, v55, v53, v46);
      v37 = v36;
      LOWORD(v36) = v47;
    }
  }
  v68 = v36;
  v69 = v37;
  v72 = sub_2D5B750(&v68);
  v73 = v38;
  if ( v61 > v72 )
    goto LABEL_42;
  v62 = v58;
  v39 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * *(unsigned int *)(a3 + 8));
  v40 = *v39;
  v41 = *((_QWORD *)v39 + 1);
  LOWORD(v72) = v40;
  v73 = v41;
  if ( !(_WORD)v40 )
  {
    v59 = v41;
    if ( !sub_30070B0((__int64)&v72) )
    {
      v67 = v59;
      v66 = 0;
      goto LABEL_56;
    }
    LOWORD(v40) = sub_3009970((__int64)&v72, v32, v59, v48, v49);
LABEL_61:
    v66 = v40;
    v67 = v51;
    if ( (_WORD)v40 )
      goto LABEL_39;
LABEL_56:
    v70 = sub_3007260((__int64)&v66);
    v71 = v50;
    if ( v62 > v70 )
      goto LABEL_42;
    goto LABEL_57;
  }
  if ( (unsigned __int16)(v40 - 17) <= 0xD3u )
  {
    LOWORD(v40) = word_4456580[v40 - 1];
    v51 = 0;
    goto LABEL_61;
  }
  v66 = v40;
  v67 = v41;
LABEL_39:
  if ( (_WORD)v40 == 1 || (unsigned __int16)(v40 - 504) <= 7u )
LABEL_70:
    BUG();
  if ( v62 > *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v40 - 16] )
  {
LABEL_42:
    *(_QWORD *)a2 = sub_383B380((__int64)a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8));
    *(_DWORD *)(a2 + 8) = v42;
    *(_QWORD *)a3 = sub_383B380((__int64)a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8));
    *(_DWORD *)(a3 + 8) = v43;
    return v43;
  }
LABEL_57:
  *(_QWORD *)a2 = v10;
  *(_DWORD *)(a2 + 8) = v9;
  *(_QWORD *)a3 = v65;
  *(_DWORD *)(a3 + 8) = v64;
  return (unsigned int)v64;
}
