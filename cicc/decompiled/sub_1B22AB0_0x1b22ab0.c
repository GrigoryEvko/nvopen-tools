// Function: sub_1B22AB0
// Address: 0x1b22ab0
//
__int64 __fastcall sub_1B22AB0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned int v9; // r13d
  __int64 v10; // r15
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  int v18; // ebx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // eax
  __int64 *v23; // r13
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // rdx
  int v28; // ecx
  unsigned __int64 v29; // r14
  __int64 *v30; // rbx
  __int64 *v31; // r13
  __int64 *v32; // rdx
  int v33; // edi
  __int64 v34; // rax
  _QWORD *v35; // rax
  int v36; // r8d
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 *v39; // r10
  __int64 v40; // r11
  __int64 v41; // rsi
  __int64 v42; // rdx
  int v43; // r8d
  __int64 v44; // rcx
  __int64 *v45; // rbx
  double v46; // xmm4_8
  double v47; // xmm5_8
  __int64 v48; // rsi
  __int64 v49; // r13
  _QWORD *v50; // rdi
  __int64 v51; // rbx
  unsigned __int64 *v52; // rcx
  unsigned __int64 v53; // rdx
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 *v56; // rbx
  __int64 *v57; // r12
  __int64 v58; // rdi
  __int64 v60; // rsi
  unsigned __int8 *v61; // rsi
  _QWORD *v62; // rax
  __int64 v63; // [rsp+8h] [rbp-188h]
  __int64 v64; // [rsp+8h] [rbp-188h]
  __int64 v65; // [rsp+8h] [rbp-188h]
  __int64 v66; // [rsp+10h] [rbp-180h]
  __int64 *v67; // [rsp+10h] [rbp-180h]
  __int64 v68; // [rsp+10h] [rbp-180h]
  __int64 v69; // [rsp+18h] [rbp-178h]
  int v70; // [rsp+28h] [rbp-168h]
  __int64 v71; // [rsp+28h] [rbp-168h]
  int v72; // [rsp+28h] [rbp-168h]
  __int64 *v73; // [rsp+30h] [rbp-160h]
  __int64 v74; // [rsp+38h] [rbp-158h]
  __int64 v75; // [rsp+40h] [rbp-150h]
  __int64 v76; // [rsp+48h] [rbp-148h]
  __int64 v77; // [rsp+58h] [rbp-138h]
  __int64 v78[2]; // [rsp+60h] [rbp-130h] BYREF
  __int16 v79; // [rsp+70h] [rbp-120h]
  __int64 *v80; // [rsp+80h] [rbp-110h] BYREF
  __int64 v81; // [rsp+88h] [rbp-108h]
  _BYTE v82[64]; // [rsp+90h] [rbp-100h] BYREF
  __int64 *v83; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v84; // [rsp+D8h] [rbp-B8h]
  _BYTE v85[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v9 = 0;
  v10 = *(_QWORD *)(a1 + 80);
  v76 = a1 + 72;
  if ( v10 != a1 + 72 )
  {
    while ( 1 )
    {
      v11 = v10 - 24;
      if ( !v10 )
        v11 = 0;
      v77 = v11;
      v12 = sub_157EBA0(v11);
      v14 = v12;
      if ( *(_BYTE *)(v12 + 16) != 29 )
        goto LABEL_44;
      if ( *(char *)(v12 + 23) >= 0 )
        goto LABEL_47;
      v15 = sub_1648A40(v12);
      v17 = v15 + v16;
      if ( *(char *)(v14 + 23) >= 0 )
        break;
      if ( !(unsigned int)((v17 - sub_1648A40(v14)) >> 4) )
        goto LABEL_47;
      if ( *(char *)(v14 + 23) >= 0 )
        goto LABEL_60;
      v18 = *(_DWORD *)(sub_1648A40(v14) + 8);
      if ( *(char *)(v14 + 23) >= 0 )
        BUG();
      v19 = sub_1648A40(v14);
      v21 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v18);
LABEL_11:
      v22 = *(_DWORD *)(v14 + 20);
      v23 = (__int64 *)(v14 + v21);
      v84 = 0x1000000000LL;
      v24 = 3LL * (v22 & 0xFFFFFFF);
      v25 = (__int64 *)v85;
      v24 *= 8;
      v26 = (__int64 *)(v14 - v24);
      v27 = v21 + v24;
      v83 = (__int64 *)v85;
      v28 = 0;
      v29 = 0xAAAAAAAAAAAAAAABLL * (v27 >> 3);
      if ( (unsigned __int64)v27 > 0x180 )
      {
        sub_16CD150((__int64)&v83, v85, 0xAAAAAAAAAAAAAAABLL * (v27 >> 3), 8, v27 >> 3, v13);
        v28 = v84;
        v25 = &v83[(unsigned int)v84];
      }
      if ( v26 != v23 )
      {
        do
        {
          if ( v25 )
            *v25 = *v26;
          v26 += 3;
          ++v25;
        }
        while ( v23 != v26 );
        v28 = v84;
      }
      v80 = (__int64 *)v82;
      LODWORD(v84) = v29 + v28;
      v81 = 0x100000000LL;
      sub_1B22560(v14, (__int64)&v80);
      v79 = 257;
      v30 = v80;
      v73 = v83;
      v74 = *(_QWORD *)(v14 - 72);
      v75 = *(_QWORD *)(*(_QWORD *)v74 + 24LL);
      v31 = &v80[7 * (unsigned int)v81];
      if ( v80 == v31 )
      {
        v65 = (unsigned int)v84;
        v68 = (unsigned int)v81;
        v72 = v84 + 1;
        v62 = sub_1648AB0(72, (int)v84 + 1, 16 * (int)v81);
        v43 = v72;
        v44 = v65;
        v37 = (__int64)v62;
        if ( v62 )
        {
          v71 = (__int64)v62;
          v38 = v65;
          v39 = v30;
          v40 = v68;
LABEL_25:
          v64 = v38;
          v67 = v39;
          v69 = v40;
          sub_15F1EA0(v37, **(_QWORD **)(v75 + 16), 54, v37 - 24 * v44 - 24, v43, v14);
          *(_QWORD *)(v37 + 56) = 0;
          sub_15F5B40(v37, v75, v74, v73, v64, (__int64)v78, v67, v69);
          goto LABEL_26;
        }
      }
      else
      {
        v32 = v80;
        v33 = 0;
        do
        {
          v34 = v32[5] - v32[4];
          v32 += 7;
          v33 += v34 >> 3;
        }
        while ( v31 != v32 );
        v63 = (unsigned int)v84;
        v66 = (unsigned int)v81;
        v70 = v84 + 1;
        v35 = sub_1648AB0(72, v33 + (int)v84 + 1, 16 * (int)v81);
        v36 = v70;
        v37 = (__int64)v35;
        if ( v35 )
        {
          v71 = (__int64)v35;
          v38 = v63;
          v39 = v30;
          v40 = v66;
          LODWORD(v41) = 0;
          do
          {
            v42 = v30[5] - v30[4];
            v30 += 7;
            v41 = (unsigned int)(v42 >> 3) + (unsigned int)v41;
          }
          while ( v31 != v30 );
          v43 = v41 + v36;
          v44 = v41 + v63;
          goto LABEL_25;
        }
      }
      v71 = 0;
      v37 = 0;
LABEL_26:
      v45 = (__int64 *)(v37 + 48);
      sub_164B7C0(v71, v14);
      *(_WORD *)(v37 + 18) = *(_WORD *)(v37 + 18) & 0x8000
                           | *(_WORD *)(v37 + 18) & 3
                           | (4 * ((*(_WORD *)(v14 + 18) >> 2) & 0xDFFF));
      *(_QWORD *)(v37 + 56) = *(_QWORD *)(v14 + 56);
      v48 = *(_QWORD *)(v14 + 48);
      v78[0] = v48;
      if ( v48 )
      {
        sub_1623A60((__int64)v78, v48, 2);
        if ( v45 != v78 )
        {
          v60 = *(_QWORD *)(v37 + 48);
          if ( v60 )
LABEL_50:
            sub_161E7C0(v37 + 48, v60);
          v61 = (unsigned __int8 *)v78[0];
          *(_QWORD *)(v37 + 48) = v78[0];
          if ( v61 )
            sub_1623210((__int64)v78, v61, v37 + 48);
          goto LABEL_30;
        }
        if ( v78[0] )
          sub_161E7C0((__int64)v78, v78[0]);
      }
      else if ( v45 != v78 )
      {
        v60 = *(_QWORD *)(v37 + 48);
        if ( v60 )
          goto LABEL_50;
      }
LABEL_30:
      sub_164D160(v14, v37, a2, a3, a4, a5, v46, v47, a8, a9);
      v49 = *(_QWORD *)(v14 - 48);
      v50 = sub_1648A60(56, 1u);
      if ( v50 )
        sub_15F8320((__int64)v50, v49, v14);
      v51 = v77;
      sub_157F2D0(*(_QWORD *)(v14 - 24), v77, 0);
      sub_157EA20(v51 + 40, v14);
      v52 = *(unsigned __int64 **)(v14 + 32);
      v53 = *(_QWORD *)(v14 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      *v52 = v53 | *v52 & 7;
      *(_QWORD *)(v53 + 8) = v52;
      *(_QWORD *)(v14 + 24) &= 7uLL;
      *(_QWORD *)(v14 + 32) = 0;
      sub_164BEC0(v14, v14, v53, (__int64)v52, a2, a3, a4, a5, v54, v55, a8, a9);
      v56 = v80;
      v57 = &v80[7 * (unsigned int)v81];
      if ( v80 != v57 )
      {
        do
        {
          v58 = *(v57 - 3);
          v57 -= 7;
          if ( v58 )
            j_j___libc_free_0(v58, v57[6] - v58);
          if ( (__int64 *)*v57 != v57 + 2 )
            j_j___libc_free_0(*v57, v57[2] + 1);
        }
        while ( v56 != v57 );
        v57 = v80;
      }
      if ( v57 != (__int64 *)v82 )
        _libc_free((unsigned __int64)v57);
      if ( v83 != (__int64 *)v85 )
        _libc_free((unsigned __int64)v83);
      v9 = 1;
LABEL_44:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v76 == v10 )
        return v9;
    }
    if ( (unsigned int)(v17 >> 4) )
LABEL_60:
      BUG();
LABEL_47:
    v21 = -72;
    goto LABEL_11;
  }
  return v9;
}
