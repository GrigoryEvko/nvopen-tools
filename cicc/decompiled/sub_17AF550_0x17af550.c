// Function: sub_17AF550
// Address: 0x17af550
//
__int64 __fastcall sub_17AF550(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r14
  __int64 v13; // rbx
  _QWORD *v14; // rax
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // rdx
  __int64 v18; // rdi
  int v19; // eax
  int v20; // r13d
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rsi
  __int64 v25; // r13
  __int64 v26; // rsi
  unsigned __int8 *v27; // rsi
  __int64 v28; // r13
  __int64 v29; // rcx
  unsigned int v30; // eax
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // rbx
  _QWORD *v34; // rax
  __int64 v35; // r15
  unsigned __int8 v36; // al
  __int64 *v37; // rbx
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rdx
  unsigned __int8 *v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  int v48; // eax
  __int64 v49; // rax
  int v50; // edx
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rcx
  unsigned __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 *v59; // rdi
  __int64 *v61; // r15
  __int64 v62; // rdx
  _QWORD *v63; // rax
  _QWORD *v64; // r15
  __int64 v65; // rsi
  __int64 v66; // rsi
  __int64 v67; // rdx
  unsigned __int8 *v68; // rsi
  __int64 v69; // rsi
  __int64 v70; // rsi
  __int64 v71; // rsi
  __int64 v72; // rdx
  unsigned __int8 *v73; // rsi
  __int64 v74; // rsi
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  int v79; // eax
  __int64 v80; // rax
  int v81; // edx
  __int64 v82; // rdx
  __int64 v83; // rsi
  unsigned __int64 v84; // rdx
  __int64 v85; // rdx
  __int64 *v86; // r13
  __int64 v87; // r14
  __int64 v88; // r15
  __int64 v89; // rbx
  _QWORD *v90; // rax
  double v91; // xmm4_8
  double v92; // xmm5_8
  __int64 v93; // rsi
  __int64 v94; // [rsp+0h] [rbp-D0h]
  __int64 v95; // [rsp+0h] [rbp-D0h]
  __int64 v96; // [rsp+10h] [rbp-C0h]
  __int64 v97; // [rsp+18h] [rbp-B8h]
  __int64 v98; // [rsp+18h] [rbp-B8h]
  _QWORD *v99; // [rsp+18h] [rbp-B8h]
  __int64 v100; // [rsp+18h] [rbp-B8h]
  __int64 v101; // [rsp+20h] [rbp-B0h]
  _QWORD *v102; // [rsp+20h] [rbp-B0h]
  __int64 *v103; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v106; // [rsp+48h] [rbp-88h] BYREF
  _QWORD v107[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v108[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v109; // [rsp+70h] [rbp-60h]
  __int64 *v110; // [rsp+80h] [rbp-50h] BYREF
  __int64 v111; // [rsp+88h] [rbp-48h]
  _BYTE v112[64]; // [rsp+90h] [rbp-40h] BYREF

  v11 = *(_QWORD *)(a3 + 8);
  v110 = (__int64 *)v112;
  v111 = 0x200000000LL;
  if ( !v11 )
    return v11;
  v13 = 0;
  do
  {
    while ( 1 )
    {
      v14 = sub_1648700(v11);
      if ( *((_BYTE *)v14 + 16) != 83 )
        break;
      if ( *(a2 - 3) != *(v14 - 3) )
      {
LABEL_52:
        v59 = v110;
        v11 = 0;
        goto LABEL_53;
      }
      v17 = (unsigned int)v111;
      if ( (unsigned int)v111 >= HIDWORD(v111) )
      {
        v102 = v14;
        sub_16CD150((__int64)&v110, v112, 0, 8, v15, v16);
        v17 = (unsigned int)v111;
        v14 = v102;
      }
      v110[v17] = (__int64)v14;
      LODWORD(v111) = v111 + 1;
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        goto LABEL_10;
    }
    if ( v13 )
      goto LABEL_52;
    v11 = *(_QWORD *)(v11 + 8);
    v13 = (__int64)v14;
  }
  while ( v11 );
LABEL_10:
  if ( v13 )
  {
    v18 = *(_QWORD *)(v13 + 8);
    if ( v18 )
    {
      if ( !*(_QWORD *)(v18 + 8)
        && (_QWORD *)a3 == sub_1648700(v18)
        && (unsigned int)*(unsigned __int8 *)(v13 + 16) - 35 <= 0x11
        && sub_17AE1C0(v13, 1u) )
      {
        v19 = *(_DWORD *)(a3 + 20);
        v109 = 257;
        v20 = v19 & 0xFFFFFFF;
        v21 = *a2;
        v22 = sub_1648B60(64);
        v23 = v22;
        if ( v22 )
        {
          sub_15F1EA0(v22, v21, 53, 0, 0, 0);
          *(_DWORD *)(v23 + 56) = v20;
          sub_164B780(v23, v108);
          sub_1648880(v23, *(_DWORD *)(v23 + 56), 1);
        }
        v24 = *(_QWORD *)(a3 + 48);
        v107[0] = v24;
        if ( v24 )
        {
          v25 = v23 + 48;
          sub_1623A60((__int64)v107, v24, 2);
          v26 = *(_QWORD *)(v23 + 48);
          if ( v26 )
            goto LABEL_20;
LABEL_21:
          v27 = (unsigned __int8 *)v107[0];
          *(_QWORD *)(v23 + 48) = v107[0];
          if ( v27 )
            sub_1623210((__int64)v107, v27, v25);
        }
        else
        {
          v26 = *(_QWORD *)(v23 + 48);
          v25 = v23 + 48;
          if ( v26 )
          {
LABEL_20:
            sub_161E7C0(v25, v26);
            goto LABEL_21;
          }
        }
        v28 = 0;
        sub_157E9D0(*(_QWORD *)(a3 + 40) + 40LL, v23);
        v29 = *(_QWORD *)(a3 + 24);
        *(_QWORD *)(v23 + 32) = a3 + 24;
        v29 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v23 + 24) = v29 | *(_QWORD *)(v23 + 24) & 7LL;
        *(_QWORD *)(v29 + 8) = v23 + 24;
        *(_QWORD *)(a3 + 24) = *(_QWORD *)(a3 + 24) & 7LL | (v23 + 24);
        sub_170B990(*a1, v23);
        v30 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
        if ( !v30 )
        {
LABEL_90:
          v59 = v110;
          v103 = &v110[(unsigned int)v111];
          if ( v103 != v110 )
          {
            v86 = v110;
            do
            {
              v87 = *v86;
              v88 = *(_QWORD *)(*v86 + 8);
              if ( v88 )
              {
                v89 = *a1;
                do
                {
                  v90 = sub_1648700(v88);
                  sub_170B990(v89, (__int64)v90);
                  v88 = *(_QWORD *)(v88 + 8);
                }
                while ( v88 );
                v93 = v23;
                if ( v23 == v87 )
                  v93 = sub_1599EF0(*(__int64 ***)v23);
                sub_164D160(v87, v93, a4, a5, a6, a7, v91, v92, a10, a11);
              }
              ++v86;
            }
            while ( v103 != v86 );
            v59 = v110;
          }
          v11 = (__int64)a2;
          goto LABEL_53;
        }
        v96 = v13;
        v31 = a3;
        while ( 2 )
        {
          if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
            v32 = *(_QWORD *)(v31 - 8);
          else
            v32 = v31 - 24LL * v30;
          v33 = *(_QWORD *)(v32 + 24 * v28);
          v101 = *(_QWORD *)(v32 + 8 * v28 + 24LL * *(unsigned int *)(v31 + 56) + 8);
          if ( v96 != v33 )
          {
            v97 = *(a2 - 3);
            v109 = 257;
            v34 = sub_1648A60(56, 2u);
            v35 = (__int64)v34;
            if ( v34 )
              sub_15FA320((__int64)v34, (_QWORD *)v33, v97, (__int64)v108, 0);
            v36 = *(_BYTE *)(v33 + 16);
            if ( v36 == 77 || v36 <= 0x17u )
            {
              v37 = (__int64 *)sub_157EE30(v101);
              if ( !v37 )
LABEL_103:
                BUG();
            }
            else
            {
              v37 = *(__int64 **)(v33 + 32);
              if ( !v37 )
                goto LABEL_103;
            }
            v38 = v37[3];
            v108[0] = v38;
            if ( v38 )
            {
              sub_1623A60((__int64)v108, v38, 2);
              v39 = *(_QWORD *)(v35 + 48);
              v40 = v35 + 48;
              if ( v39 )
                goto LABEL_34;
LABEL_35:
              v41 = (unsigned __int8 *)v108[0];
              *(_QWORD *)(v35 + 48) = v108[0];
              if ( v41 )
                sub_1623210((__int64)v108, v41, v40);
            }
            else
            {
              v39 = *(_QWORD *)(v35 + 48);
              v40 = v35 + 48;
              if ( v39 )
              {
LABEL_34:
                v98 = v40;
                sub_161E7C0(v40, v39);
                v40 = v98;
                goto LABEL_35;
              }
            }
            sub_157E9D0(v37[2] + 40, v35);
            v42 = *v37;
            v43 = *(_QWORD *)(v35 + 24);
            *(_QWORD *)(v35 + 32) = v37;
            v42 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v35 + 24) = v42 | v43 & 7;
            *(_QWORD *)(v42 + 8) = v35 + 24;
            *v37 = *v37 & 7 | (v35 + 24);
            sub_170B990(*a1, v35);
            v48 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
            if ( v48 == *(_DWORD *)(v23 + 56) )
            {
              sub_15F55D0(v23, v35, v44, v45, v46, v47);
              v48 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
            }
            v49 = (v48 + 1) & 0xFFFFFFF;
            v50 = v49 | *(_DWORD *)(v23 + 20) & 0xF0000000;
            *(_DWORD *)(v23 + 20) = v50;
            if ( (v50 & 0x40000000) != 0 )
              v51 = *(_QWORD *)(v23 - 8);
            else
              v51 = v23 - 24 * v49;
            v52 = (__int64 *)(v51 + 24LL * (unsigned int)(v49 - 1));
            if ( *v52 )
            {
              v53 = v52[1];
              v54 = v52[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v54 = v53;
              if ( v53 )
                *(_QWORD *)(v53 + 16) = *(_QWORD *)(v53 + 16) & 3LL | v54;
            }
            *v52 = v35;
            v55 = *(_QWORD *)(v35 + 8);
            v52[1] = v55;
            if ( v55 )
              *(_QWORD *)(v55 + 16) = (unsigned __int64)(v52 + 1) | *(_QWORD *)(v55 + 16) & 3LL;
            v56 = (v35 + 8) | v52[2] & 3;
LABEL_47:
            v52[2] = v56;
            *(_QWORD *)(v35 + 8) = v52;
            v57 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
              v58 = *(_QWORD *)(v23 - 8);
            else
              v58 = v23 - 24 * v57;
            ++v28;
            *(_QWORD *)(v58 + 8LL * (unsigned int)(v57 - 1) + 24LL * *(unsigned int *)(v23 + 56) + 8) = v101;
            v30 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
            if ( v30 <= (unsigned int)v28 )
              goto LABEL_90;
            continue;
          }
          break;
        }
        v94 = *(a2 - 3);
        v61 = (__int64 *)(v96
                        + 24LL * ((v31 == *(_QWORD *)(v96 - 48)) & (unsigned __int8)(*(_QWORD *)(v96 - 48) != 0))
                        - 48);
        v107[0] = sub_1649960(*v61);
        v108[0] = (__int64)v107;
        v109 = 773;
        v107[1] = v62;
        v108[1] = (__int64)".Elt";
        v99 = (_QWORD *)*v61;
        v63 = sub_1648A60(56, 2u);
        v64 = v63;
        if ( v63 )
          sub_15FA320((__int64)v63, v99, v94, (__int64)v108, 0);
        v65 = *(_QWORD *)(v33 + 48);
        v106 = (unsigned __int8 *)v65;
        if ( v65 )
        {
          sub_1623A60((__int64)&v106, v65, 2);
          v66 = v64[6];
          v67 = (__int64)(v64 + 6);
          if ( v66 )
            goto LABEL_62;
LABEL_63:
          v68 = v106;
          v64[6] = v106;
          if ( v68 )
            sub_1623210((__int64)&v106, v68, v67);
        }
        else
        {
          v66 = v64[6];
          v67 = (__int64)(v64 + 6);
          if ( v66 )
          {
LABEL_62:
            v100 = v67;
            sub_161E7C0(v67, v66);
            v67 = v100;
            goto LABEL_63;
          }
        }
        sub_157E9D0(*(_QWORD *)(v33 + 40) + 40LL, (__int64)v64);
        v69 = *(_QWORD *)(v33 + 24);
        v64[4] = v33 + 24;
        v69 &= 0xFFFFFFFFFFFFFFF8LL;
        v64[3] = v69 | v64[3] & 7LL;
        *(_QWORD *)(v69 + 8) = v64 + 3;
        *(_QWORD *)(v33 + 24) = *(_QWORD *)(v33 + 24) & 7LL | (unsigned __int64)(v64 + 3);
        sub_170B990(*a1, (__int64)v64);
        v109 = 257;
        v35 = sub_15FB440(
                (unsigned int)*(unsigned __int8 *)(v33 + 16) - 24,
                (__int64 *)v23,
                (__int64)v64,
                (__int64)v108,
                0);
        sub_15F2530((unsigned __int8 *)v35, v33, 1);
        v70 = *(_QWORD *)(v33 + 48);
        v107[0] = v70;
        if ( v70 )
        {
          sub_1623A60((__int64)v107, v70, 2);
          v71 = *(_QWORD *)(v35 + 48);
          v72 = v35 + 48;
          if ( v71 )
            goto LABEL_67;
LABEL_68:
          v73 = (unsigned __int8 *)v107[0];
          *(_QWORD *)(v35 + 48) = v107[0];
          if ( v73 )
            sub_1623210((__int64)v107, v73, v72);
        }
        else
        {
          v71 = *(_QWORD *)(v35 + 48);
          v72 = v35 + 48;
          if ( v71 )
          {
LABEL_67:
            v95 = v72;
            sub_161E7C0(v72, v71);
            v72 = v95;
            goto LABEL_68;
          }
        }
        sub_157E9D0(*(_QWORD *)(v33 + 40) + 40LL, v35);
        v74 = *(_QWORD *)(v33 + 24);
        *(_QWORD *)(v35 + 32) = v33 + 24;
        v74 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v35 + 24) = v74 | *(_QWORD *)(v35 + 24) & 7LL;
        *(_QWORD *)(v74 + 8) = v35 + 24;
        *(_QWORD *)(v33 + 24) = *(_QWORD *)(v33 + 24) & 7LL | (v35 + 24);
        sub_170B990(*a1, v35);
        v79 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        if ( v79 == *(_DWORD *)(v23 + 56) )
        {
          sub_15F55D0(v23, v35, v75, v76, v77, v78);
          v79 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        }
        v80 = (v79 + 1) & 0xFFFFFFF;
        v81 = v80 | *(_DWORD *)(v23 + 20) & 0xF0000000;
        *(_DWORD *)(v23 + 20) = v81;
        if ( (v81 & 0x40000000) != 0 )
          v82 = *(_QWORD *)(v23 - 8);
        else
          v82 = v23 - 24 * v80;
        v52 = (__int64 *)(v82 + 24LL * (unsigned int)(v80 - 1));
        if ( *v52 )
        {
          v83 = v52[1];
          v84 = v52[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v84 = v83;
          if ( v83 )
            *(_QWORD *)(v83 + 16) = *(_QWORD *)(v83 + 16) & 3LL | v84;
        }
        *v52 = v35;
        v85 = *(_QWORD *)(v35 + 8);
        v52[1] = v85;
        if ( v85 )
          *(_QWORD *)(v85 + 16) = (unsigned __int64)(v52 + 1) | *(_QWORD *)(v85 + 16) & 3LL;
        v56 = (v35 + 8) | v52[2] & 3;
        goto LABEL_47;
      }
    }
  }
  v59 = v110;
LABEL_53:
  if ( v59 != (__int64 *)v112 )
    _libc_free((unsigned __int64)v59);
  return v11;
}
