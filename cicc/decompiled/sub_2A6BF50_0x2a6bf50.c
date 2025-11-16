// Function: sub_2A6BF50
// Address: 0x2a6bf50
//
void __fastcall sub_2A6BF50(unsigned __int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 **v4; // rdx
  unsigned __int8 *v5; // rax
  unsigned __int8 *v6; // rdx
  unsigned __int8 *v7; // rax
  __int64 v8; // rdi
  int v9; // edx
  unsigned int v10; // eax
  unsigned __int8 v11; // dl
  unsigned int v12; // esi
  unsigned int v13; // eax
  unsigned __int8 v14; // dl
  unsigned int v15; // esi
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  int v18; // edx
  __int32 v19; // eax
  unsigned int v20; // eax
  unsigned __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // r8
  char v24; // al
  __int64 *v25; // r8
  unsigned __int8 *v26; // rdx
  __int64 *v27; // rdx
  unsigned __int64 v28; // rax
  int v29; // edi
  unsigned __int8 *v30; // rax
  unsigned __int8 *v31; // rdx
  __int64 v32; // rax
  unsigned __int8 *v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rcx
  __int64 *v36; // rax
  __int64 v37; // [rsp+28h] [rbp-198h]
  unsigned __int8 v38; // [rsp+38h] [rbp-188h]
  unsigned int v39; // [rsp+38h] [rbp-188h]
  unsigned __int8 v40; // [rsp+40h] [rbp-180h]
  __int64 *v41; // [rsp+40h] [rbp-180h]
  unsigned int v42; // [rsp+40h] [rbp-180h]
  __int64 *v43; // [rsp+48h] [rbp-178h]
  __int64 v44; // [rsp+50h] [rbp-170h] BYREF
  unsigned int v45; // [rsp+58h] [rbp-168h]
  __int64 v46; // [rsp+60h] [rbp-160h] BYREF
  unsigned int v47; // [rsp+68h] [rbp-158h]
  __int64 v48; // [rsp+70h] [rbp-150h] BYREF
  unsigned int v49; // [rsp+78h] [rbp-148h]
  __int64 v50; // [rsp+80h] [rbp-140h] BYREF
  unsigned int v51; // [rsp+88h] [rbp-138h]
  unsigned __int64 v52; // [rsp+90h] [rbp-130h] BYREF
  unsigned int v53; // [rsp+98h] [rbp-128h]
  unsigned __int64 v54; // [rsp+A0h] [rbp-120h] BYREF
  unsigned int v55; // [rsp+A8h] [rbp-118h]
  unsigned __int8 v56[8]; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v57; // [rsp+B8h] [rbp-108h] BYREF
  unsigned int v58; // [rsp+C0h] [rbp-100h]
  __int64 v59; // [rsp+C8h] [rbp-F8h] BYREF
  unsigned int v60; // [rsp+D0h] [rbp-F0h]
  unsigned __int8 v61[8]; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v62; // [rsp+E8h] [rbp-D8h] BYREF
  unsigned int v63; // [rsp+F0h] [rbp-D0h]
  __int64 v64; // [rsp+F8h] [rbp-C8h] BYREF
  unsigned int v65; // [rsp+100h] [rbp-C0h]
  unsigned __int64 v66; // [rsp+110h] [rbp-B0h] BYREF
  unsigned int v67; // [rsp+118h] [rbp-A8h]
  unsigned __int64 v68; // [rsp+120h] [rbp-A0h] BYREF
  unsigned int v69; // [rsp+128h] [rbp-98h]
  __m128i v70; // [rsp+140h] [rbp-80h] BYREF
  unsigned __int64 v71; // [rsp+150h] [rbp-70h] BYREF
  __int64 v72; // [rsp+158h] [rbp-68h]
  __int64 v73; // [rsp+160h] [rbp-60h]
  unsigned __int8 *v74; // [rsp+168h] [rbp-58h]
  __int64 v75; // [rsp+170h] [rbp-50h]
  __int64 v76; // [rsp+178h] [rbp-48h]
  __int16 v77; // [rsp+180h] [rbp-40h]

  if ( (a2[7] & 0x40) != 0 )
    v4 = (unsigned __int8 **)*((_QWORD *)a2 - 1);
  else
    v4 = (unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v5 = (unsigned __int8 *)sub_2A68BC0((__int64)a1, *v4);
  sub_22C05A0((__int64)v56, v5);
  if ( (a2[7] & 0x40) != 0 )
    v6 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v6 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v7 = (unsigned __int8 *)sub_2A68BC0((__int64)a1, *((unsigned __int8 **)v6 + 4));
  sub_22C05A0((__int64)v61, v7);
  v70.m128i_i64[0] = (__int64)a2;
  v43 = sub_2A686D0((__int64)(a1 + 17), v70.m128i_i64);
  if ( *(_BYTE *)v43 == 6 || v56[0] <= 1u || v61[0] <= 1u )
    goto LABEL_38;
  if ( v56[0] == 6 && v61[0] == 6 )
    goto LABEL_37;
  if ( v56[0] == 2 || v61[0] == 2 )
  {
    if ( (unsigned __int8)sub_2A62D90((__int64)v56) )
    {
      v33 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v23 = sub_2A637C0((__int64)a1, (__int64)v56, *(_QWORD *)(*(_QWORD *)v33 + 8LL));
    }
    else
    {
      v22 = (a2[7] & 0x40) != 0
          ? (__int64 *)*((_QWORD *)a2 - 1)
          : (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v23 = *v22;
    }
    v41 = (__int64 *)v23;
    v24 = sub_2A62D90((__int64)v61);
    v25 = v41;
    if ( v24 )
    {
      v31 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v32 = sub_2A637C0((__int64)a1, (__int64)v61, *(_QWORD *)(*((_QWORD *)v31 + 4) + 8LL));
      v25 = v41;
      v27 = (__int64 *)v32;
    }
    else
    {
      v26 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v27 = (__int64 *)*((_QWORD *)v26 + 4);
    }
    v28 = *a1;
    v29 = *a2;
    v70 = (__m128i)v28;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = a2;
    v75 = 0;
    v76 = 0;
    v77 = 257;
    v30 = sub_101E7C0(v29 - 29, v25, v27, &v70);
    if ( v30 )
    {
      if ( *v30 <= 0x15u )
      {
        v66 = 0;
        sub_2A624B0((__int64)&v66, v30, 1);
        sub_22C05A0((__int64)&v70, (unsigned __int8 *)&v66);
        sub_2A689D0((__int64)a1, (__int64)a2, (unsigned __int8 *)&v70, 0x100000000LL);
        sub_22C0090((unsigned __int8 *)&v70);
        sub_22C0090((unsigned __int8 *)&v66);
        goto LABEL_38;
      }
    }
  }
  v8 = *((_QWORD *)a2 + 1);
  v9 = *(unsigned __int8 *)(v8 + 8);
  if ( (unsigned int)(v9 - 17) <= 1 )
    LOBYTE(v9) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
  if ( (_BYTE)v9 != 12 )
  {
LABEL_37:
    sub_2A6A450((__int64)a1, (__int64)a2);
LABEL_38:
    sub_22C0090(v61);
    sub_22C0090(v56);
    return;
  }
  v40 = v56[0];
  if ( v56[0] == 4
    || (v10 = sub_BCB060(v8), v11 = v40, v12 = v10, v40 == 5)
    && (v42 = v10, v34 = sub_9876C0(&v57), v11 = v56[0], v12 = v42, v34) )
  {
    v45 = v58;
    if ( v58 > 0x40 )
      sub_C43780((__int64)&v44, (const void **)&v57);
    else
      v44 = v57;
    v47 = v60;
    if ( v60 > 0x40 )
      sub_C43780((__int64)&v46, (const void **)&v59);
    else
      v46 = v59;
  }
  else if ( v11 == 2 )
  {
    sub_AD8380((__int64)&v44, v57);
  }
  else if ( v11 )
  {
    sub_AADB10((__int64)&v44, v12, 1);
  }
  else
  {
    sub_AADB10((__int64)&v44, v12, 0);
  }
  v38 = v61[0];
  if ( v61[0] == 4
    || (v13 = sub_BCB060(*((_QWORD *)a2 + 1)), v14 = v38, v15 = v13, v38 == 5)
    && (v39 = v13, v36 = sub_9876C0(&v62), v14 = v61[0], v15 = v39, v36) )
  {
    v49 = v63;
    if ( v63 > 0x40 )
      sub_C43780((__int64)&v48, (const void **)&v62);
    else
      v48 = v62;
    v51 = v65;
    if ( v65 > 0x40 )
      sub_C43780((__int64)&v50, (const void **)&v64);
    else
      v50 = v64;
  }
  else if ( v14 == 2 )
  {
    sub_AD8380((__int64)&v48, v62);
  }
  else if ( v14 )
  {
    sub_AADB10((__int64)&v48, v15, 1);
  }
  else
  {
    sub_AADB10((__int64)&v48, v15, 0);
  }
  v16 = sub_BCB060(*((_QWORD *)a2 + 1));
  sub_AADB10((__int64)&v52, v16, 0);
  v17 = *a2;
  v18 = v17 - 29;
  if ( (unsigned __int8)v17 <= 0x36u
    && (v35 = 0x40540000000000LL, v18 = (unsigned __int8)v17 - 29, _bittest64(&v35, v17)) )
  {
    sub_ABCBD0((__int64)&v70, (__int64)&v44, v18, &v48, (a2[1] >> 1) & 3);
    if ( v53 > 0x40 )
    {
LABEL_27:
      if ( v52 )
        j_j___libc_free_0_0(v52);
    }
  }
  else
  {
    sub_ABCAA0((__int64)&v70, (__int64)&v44, v18, &v48);
    if ( v53 > 0x40 )
      goto LABEL_27;
  }
  v52 = v70.m128i_i64[0];
  v19 = v70.m128i_i32[2];
  v70.m128i_i32[2] = 0;
  v53 = v19;
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  v54 = v71;
  v20 = v72;
  LODWORD(v72) = 0;
  v55 = v20;
  sub_969240((__int64 *)&v71);
  sub_969240(v70.m128i_i64);
  v21 = (unsigned __int64)(unsigned int)qword_500BEC8 << 32;
  BYTE1(v21) = 1;
  v37 = v21;
  v67 = v53;
  if ( v53 > 0x40 )
    sub_C43780((__int64)&v66, (const void **)&v52);
  else
    v66 = v52;
  v69 = v55;
  if ( v55 > 0x40 )
    sub_C43780((__int64)&v68, (const void **)&v54);
  else
    v68 = v54;
  sub_22C06B0((__int64)&v70, (__int64)&v66, 0);
  sub_2A639B0((__int64)a1, v43, (__int64)a2, (__int64)&v70, v37);
  sub_22C0090((unsigned __int8 *)&v70);
  sub_969240((__int64 *)&v68);
  sub_969240((__int64 *)&v66);
  sub_969240((__int64 *)&v54);
  sub_969240((__int64 *)&v52);
  sub_969240(&v50);
  sub_969240(&v48);
  sub_969240(&v46);
  sub_969240(&v44);
  sub_22C0090(v61);
  sub_22C0090(v56);
}
