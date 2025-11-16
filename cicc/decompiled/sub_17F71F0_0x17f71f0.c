// Function: sub_17F71F0
// Address: 0x17f71f0
//
__int64 *__fastcall sub_17F71F0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 *result; // rax
  __int64 **v5; // rax
  unsigned int v6; // eax
  __int64 v7; // r14
  unsigned __int8 *v8; // rsi
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int8 *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rbx
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rbx
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 **v23; // r14
  __int64 v24; // r14
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 *v27; // rbx
  unsigned int v28; // r15d
  __int64 **v29; // r13
  int v30; // r8d
  int v31; // r9d
  __int64 v32; // rax
  __int64 v33; // rsi
  unsigned __int8 *v34; // rbx
  unsigned __int64 v35; // r14
  __int64 *v36; // r12
  unsigned __int64 v37; // rax
  __int64 *v38; // r13
  __int64 v39; // r12
  unsigned __int64 v40; // r15
  unsigned int v41; // r14d
  unsigned __int64 v42; // rax
  unsigned int v43; // r14d
  __int64 v44; // rbx
  unsigned int v45; // r14d
  __int64 *v46; // rax
  __int128 v47; // rdi
  __int64 v48; // r15
  __int64 v49; // rcx
  __int64 v50; // rbx
  __int64 v51; // r14
  __int64 v52; // rsi
  __int64 v53; // rdx
  __int64 *v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rsi
  unsigned __int8 *v58; // rsi
  __int64 *v59; // rbx
  __int64 v60; // rax
  __int64 v61; // rcx
  __int64 v62; // rbx
  __int64 v63; // rsi
  unsigned __int8 *v64; // rsi
  _QWORD *v65; // [rsp+8h] [rbp-1D8h]
  __int64 *v66; // [rsp+18h] [rbp-1C8h]
  __int64 *v67; // [rsp+28h] [rbp-1B8h]
  __int64 v68; // [rsp+30h] [rbp-1B0h]
  __int64 v69; // [rsp+38h] [rbp-1A8h]
  __int64 *v70; // [rsp+38h] [rbp-1A8h]
  _QWORD *v71; // [rsp+40h] [rbp-1A0h]
  __int64 v72; // [rsp+40h] [rbp-1A0h]
  __int64 *v73; // [rsp+48h] [rbp-198h]
  unsigned __int8 *v74; // [rsp+58h] [rbp-188h] BYREF
  _QWORD v75[2]; // [rsp+60h] [rbp-180h] BYREF
  __int64 v76[2]; // [rsp+70h] [rbp-170h] BYREF
  __int16 v77; // [rsp+80h] [rbp-160h]
  __int64 v78[2]; // [rsp+90h] [rbp-150h] BYREF
  __int16 v79; // [rsp+A0h] [rbp-140h]
  _QWORD v80[2]; // [rsp+B0h] [rbp-130h] BYREF
  __int16 v81; // [rsp+C0h] [rbp-120h]
  unsigned __int8 *v82; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v83; // [rsp+D8h] [rbp-108h]
  __int64 *v84; // [rsp+E0h] [rbp-100h]
  __int64 v85; // [rsp+E8h] [rbp-F8h]
  __int64 v86; // [rsp+F0h] [rbp-F0h]
  int v87; // [rsp+F8h] [rbp-E8h]
  __int64 v88; // [rsp+100h] [rbp-E0h]
  __int64 v89; // [rsp+108h] [rbp-D8h]
  unsigned __int8 *v90; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v91; // [rsp+128h] [rbp-B8h]
  _BYTE v92[176]; // [rsp+130h] [rbp-B0h] BYREF

  result = &a2[a3];
  v67 = result;
  v73 = a2;
  if ( a2 != result )
  {
    while ( 1 )
    {
      v9 = *v73;
      if ( *(_BYTE *)(*v73 + 16) != 27 )
        goto LABEL_7;
      v10 = sub_16498A0(*v73);
      v88 = 0;
      v89 = 0;
      v11 = *(unsigned __int8 **)(v9 + 48);
      v85 = v10;
      v87 = 0;
      v12 = *(_QWORD *)(v9 + 40);
      v82 = 0;
      v83 = v12;
      v86 = 0;
      v84 = (__int64 *)(v9 + 24);
      v90 = v11;
      if ( v11 )
      {
        sub_1623A60((__int64)&v90, (__int64)v11, 2);
        if ( v82 )
          sub_161E7C0((__int64)&v82, (__int64)v82);
        v82 = v90;
        if ( v90 )
          sub_1623210((__int64)&v90, v90, (__int64)&v82);
      }
      v90 = v92;
      v91 = 0x1000000000LL;
      v5 = (*(_BYTE *)(v9 + 23) & 0x40) != 0
         ? *(__int64 ***)(v9 - 8)
         : (__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v68 = (__int64)*v5;
      v6 = sub_16431D0(**v5);
      v7 = a1[39];
      if ( v6 <= (unsigned int)sub_16431D0(v7) )
        break;
      v8 = v82;
      if ( v82 )
        goto LABEL_6;
LABEL_7:
      result = ++v73;
      if ( v67 == v73 )
        return result;
    }
    v13 = sub_15A0680(v7, ((*(_DWORD *)(v9 + 20) & 0xFFFFFFFu) >> 1) - 1, 0);
    v16 = (unsigned int)v91;
    if ( (unsigned int)v91 >= HIDWORD(v91) )
    {
      sub_16CD150((__int64)&v90, v92, 0, 8, v14, v15);
      v16 = (unsigned int)v91;
    }
    *(_QWORD *)&v90[8 * v16] = v13;
    LODWORD(v91) = v91 + 1;
    v17 = sub_16431D0(*(_QWORD *)v68);
    v18 = sub_15A0680(a1[39], v17, 0);
    v21 = (unsigned int)v91;
    if ( (unsigned int)v91 >= HIDWORD(v91) )
    {
      sub_16CD150((__int64)&v90, v92, 0, 8, v19, v20);
      v21 = (unsigned int)v91;
    }
    *(_QWORD *)&v90[8 * v21] = v18;
    LODWORD(v91) = v91 + 1;
    v22 = sub_16431D0(*(_QWORD *)v68);
    v23 = (__int64 **)a1[39];
    if ( v22 < (unsigned int)sub_16431D0((__int64)v23) )
    {
      v79 = 257;
      if ( v23 != *(__int64 ***)v68 )
      {
        if ( *(_BYTE *)(v68 + 16) > 0x10u )
        {
          v81 = 257;
          v68 = sub_15FE0A0((_QWORD *)v68, (__int64)v23, 0, (__int64)v80, 0);
          if ( v83 )
          {
            v59 = v84;
            sub_157E9D0(v83 + 40, v68);
            v60 = *(_QWORD *)(v68 + 24);
            v61 = *v59;
            *(_QWORD *)(v68 + 32) = v59;
            v61 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v68 + 24) = v61 | v60 & 7;
            *(_QWORD *)(v61 + 8) = v68 + 24;
            *v59 = *v59 & 7 | (v68 + 24);
          }
          sub_164B780(v68, v78);
          if ( v82 )
          {
            v76[0] = (__int64)v82;
            sub_1623A60((__int64)v76, (__int64)v82, 2);
            v62 = v68 + 48;
            v63 = *(_QWORD *)(v68 + 48);
            if ( v63 )
              sub_161E7C0(v62, v63);
            v64 = (unsigned __int8 *)v76[0];
            *(_QWORD *)(v68 + 48) = v76[0];
            if ( v64 )
              sub_1623210((__int64)v76, v64, v62);
          }
        }
        else
        {
          v68 = sub_15A4750((__int64 ***)v68, v23, 0);
        }
      }
    }
    v69 = ((*(_DWORD *)(v9 + 20) & 0xFFFFFFFu) >> 1) - 1;
    if ( (*(_DWORD *)(v9 + 20) & 0xFFFFFFFu) >> 1 != 1 )
    {
      v71 = a1;
      v24 = 0;
      v25 = v9;
      while ( 1 )
      {
        ++v24;
        if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
          v26 = *(_QWORD *)(v25 - 8);
        else
          v26 = v25 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF);
        v27 = *(__int64 **)(v26 + 24LL * (unsigned int)(2 * v24));
        v28 = sub_16431D0(*v27);
        v29 = (__int64 **)v71[39];
        if ( v28 < (unsigned int)sub_16431D0((__int64)v29) )
        {
          v27 = (__int64 *)sub_15A46C0(37, (__int64 ***)v27, v29, 0);
          v32 = (unsigned int)v91;
          if ( (unsigned int)v91 >= HIDWORD(v91) )
          {
LABEL_30:
            sub_16CD150((__int64)&v90, v92, 0, 8, v30, v31);
            v32 = (unsigned int)v91;
          }
        }
        else
        {
          v32 = (unsigned int)v91;
          if ( (unsigned int)v91 >= HIDWORD(v91) )
            goto LABEL_30;
        }
        *(_QWORD *)&v90[8 * v32] = v27;
        v33 = (unsigned int)(v91 + 1);
        LODWORD(v91) = v91 + 1;
        if ( v69 == v24 )
        {
          a1 = v71;
          goto LABEL_32;
        }
      }
    }
    v33 = (unsigned int)v91;
LABEL_32:
    v34 = v90;
    v35 = 8 * v33;
    v36 = (__int64 *)(v90 + 16);
    v66 = (__int64 *)&v90[8 * v33];
    if ( v66 != (__int64 *)(v90 + 16) )
    {
      _BitScanReverse64(&v37, (__int64)(v35 - 16) >> 3);
      sub_17F4AE0((_QWORD *)v90 + 2, (char *)&v90[8 * v33], 2LL * (int)(63 - (v37 ^ 0x3F)));
      if ( v35 > 0x90 )
      {
        v70 = (__int64 *)(v34 + 144);
        sub_17F4910(v36, (__int64 *)v34 + 18);
        if ( v66 == (__int64 *)(v34 + 144) )
          goto LABEL_49;
        v65 = a1;
        while ( 1 )
        {
          v38 = v70;
          v72 = *v70 + 24;
          v39 = *v70;
          while ( 1 )
          {
            v43 = *(_DWORD *)(v39 + 32);
            v44 = *(v38 - 1);
            if ( v43 <= 0x40 )
            {
              v40 = *(_QWORD *)(v39 + 24);
LABEL_38:
              v41 = *(_DWORD *)(v44 + 32);
              if ( v41 <= 0x40 )
                goto LABEL_39;
              goto LABEL_45;
            }
            v40 = -1;
            if ( v43 - (unsigned int)sub_16A57B0(v72) > 0x40 )
              goto LABEL_38;
            v41 = *(_DWORD *)(v44 + 32);
            v40 = **(_QWORD **)(v39 + 24);
            if ( v41 <= 0x40 )
            {
LABEL_39:
              v42 = *(_QWORD *)(v44 + 24);
              goto LABEL_40;
            }
LABEL_45:
            v45 = v41 - sub_16A57B0(v44 + 24);
            v42 = -1;
            if ( v45 <= 0x40 )
              break;
LABEL_40:
            if ( v42 <= v40 )
              goto LABEL_47;
LABEL_41:
            *v38-- = v44;
          }
          if ( **(_QWORD **)(v44 + 24) > v40 )
            goto LABEL_41;
LABEL_47:
          ++v70;
          *v38 = v39;
          if ( v66 == v70 )
          {
            a1 = v65;
            goto LABEL_49;
          }
        }
      }
      sub_17F4910(v36, v66);
LABEL_49:
      v33 = (unsigned int)v91;
    }
    v46 = sub_1645D80((__int64 *)a1[39], v33);
    *((_QWORD *)&v47 + 1) = v90;
    *(_QWORD *)&v47 = v46;
    v48 = (__int64)v46;
    v50 = sub_159DFD0(v47, (unsigned int)v91, v49);
    v81 = 259;
    v80[0] = "__sancov_gen_cov_switch_values";
    v51 = (__int64)sub_1648A60(88, 1u);
    if ( v51 )
      sub_15E51E0(v51, a1[46], v48, 0, 7, v50, (__int64)v80, 0, 0, 0, 0);
    v79 = 257;
    v52 = a1[40];
    v75[0] = v68;
    v77 = 257;
    if ( v52 != *(_QWORD *)v51 )
    {
      if ( *(_BYTE *)(v51 + 16) > 0x10u )
      {
        v81 = 257;
        v51 = sub_15FDFF0(v51, v52, (__int64)v80, 0);
        if ( v83 )
        {
          v54 = v84;
          sub_157E9D0(v83 + 40, v51);
          v55 = *(_QWORD *)(v51 + 24);
          v56 = *v54;
          *(_QWORD *)(v51 + 32) = v54;
          v56 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v51 + 24) = v56 | v55 & 7;
          *(_QWORD *)(v56 + 8) = v51 + 24;
          *v54 = *v54 & 7 | (v51 + 24);
        }
        sub_164B780(v51, v76);
        if ( v82 )
        {
          v74 = v82;
          sub_1623A60((__int64)&v74, (__int64)v82, 2);
          v57 = *(_QWORD *)(v51 + 48);
          if ( v57 )
            sub_161E7C0(v51 + 48, v57);
          v58 = v74;
          *(_QWORD *)(v51 + 48) = v74;
          if ( v58 )
            sub_1623210((__int64)&v74, v58, v51 + 48);
        }
      }
      else
      {
        v51 = sub_15A4A70((__int64 ***)v51, v52);
      }
    }
    v53 = a1[34];
    v75[1] = v51;
    sub_1285290((__int64 *)&v82, *(_QWORD *)(v53 + 24), v53, (int)v75, 2, (__int64)v78, 0);
    if ( v90 != v92 )
      _libc_free((unsigned __int64)v90);
    v8 = v82;
    if ( !v82 )
      goto LABEL_7;
LABEL_6:
    sub_161E7C0((__int64)&v82, (__int64)v8);
    goto LABEL_7;
  }
  return result;
}
