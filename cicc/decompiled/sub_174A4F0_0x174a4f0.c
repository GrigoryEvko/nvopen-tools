// Function: sub_174A4F0
// Address: 0x174a4f0
//
__int64 __fastcall sub_174A4F0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  unsigned int v4; // r15d
  __int64 v6; // r12
  __int64 v10; // rcx
  int v11; // r8d
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // rax
  __int64 v16; // r12
  unsigned int v17; // eax
  unsigned int v18; // edx
  unsigned int v19; // r8d
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  _BYTE *v25; // rdi
  unsigned __int8 v26; // al
  _BYTE *v27; // rdx
  unsigned int v28; // eax
  unsigned __int64 *v29; // rdx
  __int64 v30; // r9
  unsigned __int64 v31; // r8
  unsigned int v32; // ecx
  _QWORD *v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rax
  _BYTE *v36; // rdi
  unsigned __int8 v37; // al
  _BYTE *v38; // rdx
  unsigned int v39; // eax
  unsigned __int64 *v40; // rdx
  __int64 v41; // r9
  unsigned __int64 v42; // rsi
  unsigned int v43; // r8d
  _QWORD *v44; // rax
  unsigned int v45; // eax
  unsigned int v46; // r8d
  unsigned int v47; // edx
  __int64 *v48; // rax
  __int64 *v49; // r12
  __int64 v50; // rax
  _BYTE *v51; // rdi
  unsigned __int8 v52; // al
  unsigned int v53; // eax
  _QWORD **v54; // rsi
  unsigned __int64 v55; // rdx
  _QWORD *v56; // rax
  __int64 v57; // rax
  _QWORD *v58; // rdx
  _QWORD *v59; // r8
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // rax
  __int64 v64; // rax
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // [rsp+8h] [rbp-68h]
  unsigned int v68; // [rsp+8h] [rbp-68h]
  unsigned int v69; // [rsp+10h] [rbp-60h]
  __int64 v70; // [rsp+10h] [rbp-60h]
  unsigned __int64 v71; // [rsp+18h] [rbp-58h]
  unsigned __int64 v72; // [rsp+18h] [rbp-58h]
  unsigned __int64 v73; // [rsp+18h] [rbp-58h]
  __int64 v74; // [rsp+20h] [rbp-50h]
  __int64 v75; // [rsp+20h] [rbp-50h]
  unsigned int v76; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v77; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v78; // [rsp+20h] [rbp-50h]
  unsigned int v79; // [rsp+28h] [rbp-48h]
  _BYTE *v80; // [rsp+28h] [rbp-48h]
  unsigned int v81; // [rsp+28h] [rbp-48h]
  unsigned int v82; // [rsp+28h] [rbp-48h]
  _BYTE *v83; // [rsp+28h] [rbp-48h]
  unsigned int v84; // [rsp+28h] [rbp-48h]
  unsigned int v85; // [rsp+28h] [rbp-48h]
  unsigned int v86; // [rsp+28h] [rbp-48h]
  _QWORD *v87; // [rsp+28h] [rbp-48h]
  __int64 v88; // [rsp+28h] [rbp-48h]
  __int64 v89; // [rsp+28h] [rbp-48h]
  __int64 v90; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v91; // [rsp+38h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
    return 1;
  v6 = a1;
  while ( 2 )
  {
    v4 = sub_1749B70(v6, a2);
    if ( (_BYTE)v4 )
      return 1;
    if ( (unsigned __int8)v11 <= 0x17u )
      return v4;
    v12 = *(_QWORD *)(v6 + 8);
    if ( !v12 || *(_QWORD *)(v12 + 8) )
      return v4;
    v13 = (unsigned int)(v11 - 35);
    v14 = *(_QWORD *)v6;
    switch ( (int)v13 )
    {
      case 0:
      case 2:
      case 4:
      case 15:
      case 16:
      case 17:
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v15 = *(_QWORD **)(v6 - 8);
        else
          v15 = (_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        if ( !(unsigned __int8)sub_174A4F0(*v15, a2, a3, a4, v13, v14) )
          return v4;
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v16 = *(_QWORD *)(v6 - 8);
        else
          v16 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        v6 = *(_QWORD *)(v16 + 24);
        goto LABEL_16;
      case 6:
      case 9:
        v79 = sub_16431D0(*(_QWORD *)v6);
        v17 = sub_16431D0(a2);
        v18 = v79;
        v19 = v17;
        v91 = v79;
        if ( v79 > 0x40 )
        {
          v76 = v17;
          sub_16A4EF0((__int64)&v90, 0, 0);
          v18 = v91;
          v19 = v76;
        }
        else
        {
          v90 = 0;
        }
        if ( v19 != v18 )
        {
          if ( v19 > 0x3F || v18 > 0x40 )
            sub_16A5260(&v90, v19, v18);
          else
            v90 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v19 + 64 - (unsigned __int8)v18) << v19;
        }
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v20 = *(__int64 **)(v6 - 8);
        else
          v20 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        if ( (unsigned __int8)sub_14C1670(*v20, (__int64)&v90, a3[333], 0, a3[330], a4, a3[332]) )
        {
          v21 = (*(_BYTE *)(v6 + 23) & 0x40) != 0 ? *(_QWORD *)(v6 - 8) : v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
          if ( (unsigned __int8)sub_14C1670(*(_QWORD *)(v21 + 24), (__int64)&v90, a3[333], 0, a3[330], a4, a3[332]) )
          {
            v63 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
                ? *(_QWORD **)(v6 - 8)
                : (_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
            v4 = sub_174A4F0(*v63, a2, a3, a4, v22, v23);
            if ( (_BYTE)v4 )
            {
              v64 = sub_13CF970(v6);
              v4 = sub_174A4F0(*(_QWORD *)(v64 + 24), a2, a3, a4, v65, v66);
            }
          }
        }
        goto LABEL_30;
      case 12:
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v50 = *(_QWORD *)(v6 - 8);
        else
          v50 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        v51 = *(_BYTE **)(v50 + 24);
        v52 = v51[16];
        if ( v52 != 13 )
        {
          if ( *(_BYTE *)(*(_QWORD *)v51 + 8LL) != 16 )
            return v4;
          if ( v52 > 0x10u )
            return v4;
          v60 = sub_15A1020(v51, a2, *(_QWORD *)v51, v10);
          v51 = (_BYTE *)v60;
          if ( !v60 || *(_BYTE *)(v60 + 16) != 13 )
            return v4;
        }
        v53 = sub_16431D0(a2);
        v54 = (_QWORD **)(v51 + 24);
        v55 = v53;
        v86 = *((_DWORD *)v51 + 8);
        if ( v86 > 0x40 )
        {
          v73 = v53;
          if ( v86 - (unsigned int)sub_16A57B0((__int64)v54) > 0x40 )
            return v4;
          v55 = v73;
          v56 = (_QWORD *)**v54;
          if ( v73 < (unsigned __int64)v56 )
            return v4;
        }
        else
        {
          v56 = *v54;
          if ( v55 < (unsigned __int64)*v54 )
            return v4;
        }
        if ( v55 <= (unsigned __int64)v56 )
          return v4;
        goto LABEL_59;
      case 13:
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v35 = *(_QWORD *)(v6 - 8);
        else
          v35 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        v36 = *(_BYTE **)(v35 + 24);
        v37 = v36[16];
        v38 = v36 + 24;
        if ( v37 == 13 )
          goto LABEL_46;
        if ( *(_BYTE *)(*(_QWORD *)v36 + 8LL) != 16 )
          return v4;
        v89 = *(_QWORD *)v6;
        if ( v37 > 0x10u )
          return v4;
        v62 = sub_15A1020(v36, a2, *(_QWORD *)v36, v10);
        if ( !v62 || *(_BYTE *)(v62 + 16) != 13 )
          return v4;
        v14 = v89;
        v38 = (_BYTE *)(v62 + 24);
LABEL_46:
        v75 = v14;
        v83 = v38;
        v39 = sub_16431D0(a2);
        v40 = (unsigned __int64 *)v83;
        v41 = v75;
        v42 = v39;
        v43 = v39;
        v84 = *((_DWORD *)v83 + 2);
        if ( v84 > 0x40 )
        {
          v68 = v39;
          v70 = v75;
          v72 = v39;
          v78 = v40;
          if ( v84 - (unsigned int)sub_16A57B0((__int64)v40) > 0x40 )
            return v4;
          v42 = v72;
          v41 = v70;
          v43 = v68;
          v44 = *(_QWORD **)*v78;
          if ( v72 < (unsigned __int64)v44 )
            return v4;
        }
        else
        {
          v44 = (_QWORD *)*v40;
          if ( v42 < *v40 )
            return v4;
        }
        v85 = v43;
        if ( v42 <= (unsigned __int64)v44 )
          return v4;
        v45 = sub_16431D0(v41);
        v46 = v85;
        v91 = v45;
        v47 = v45;
        if ( v45 > 0x40 )
        {
          sub_16A4EF0((__int64)&v90, 0, 0);
          v47 = v91;
          v46 = v85;
        }
        else
        {
          v90 = 0;
        }
        if ( v46 != v47 )
        {
          if ( v46 > 0x3F || v47 > 0x40 )
            sub_16A5260(&v90, v46, v47);
          else
            v90 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v46 + 64 - (unsigned __int8)v47) << v46;
        }
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v48 = *(__int64 **)(v6 - 8);
        else
          v48 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        if ( (unsigned __int8)sub_14C1670(*v48, (__int64)&v90, a3[333], 0, a3[330], a4, a3[332]) )
        {
          if ( v91 > 0x40 && v90 )
            j_j___libc_free_0_0(v90);
LABEL_59:
          if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
            v49 = *(__int64 **)(v6 - 8);
          else
            v49 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
          v6 = *v49;
LABEL_16:
          if ( *(_BYTE *)(v6 + 16) <= 0x10u )
            return 1;
          continue;
        }
LABEL_30:
        if ( v91 > 0x40 && v90 )
          j_j___libc_free_0_0(v90);
        return v4;
      case 14:
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v24 = *(_QWORD *)(v6 - 8);
        else
          v24 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        v25 = *(_BYTE **)(v24 + 24);
        v26 = v25[16];
        v27 = v25 + 24;
        if ( v26 == 13 )
          goto LABEL_36;
        if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) != 16 )
          return v4;
        v88 = *(_QWORD *)v6;
        if ( v26 > 0x10u )
          return v4;
        v61 = sub_15A1020(v25, a2, *(_QWORD *)v25, v10);
        if ( !v61 || *(_BYTE *)(v61 + 16) != 13 )
          return v4;
        v14 = v88;
        v27 = (_BYTE *)(v61 + 24);
LABEL_36:
        v74 = v14;
        v80 = v27;
        v28 = sub_16431D0(a2);
        v29 = (unsigned __int64 *)v80;
        v30 = v74;
        v31 = v28;
        v32 = v28;
        v81 = *((_DWORD *)v80 + 2);
        if ( v81 > 0x40 )
        {
          v67 = v74;
          v69 = v28;
          v71 = v28;
          v77 = v29;
          if ( v81 - (unsigned int)sub_16A57B0((__int64)v29) > 0x40 )
            return v4;
          v31 = v71;
          v32 = v69;
          v30 = v67;
          v33 = *(_QWORD **)*v77;
          if ( v71 < (unsigned __int64)v33 )
            return v4;
        }
        else
        {
          v33 = (_QWORD *)*v29;
          if ( v31 < *v29 )
            return v4;
        }
        if ( v31 <= (unsigned __int64)v33 )
          return v4;
        v82 = sub_16431D0(v30) - v32;
        v34 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
            ? *(__int64 **)(v6 - 8)
            : (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        if ( v82 >= (unsigned int)sub_14C23D0(*v34, a3[333], 0, a3[330], a4, a3[332]) )
          return v4;
        v6 = *(_QWORD *)sub_13CF970(v6);
        goto LABEL_16;
      case 25:
      case 26:
      case 27:
        return 1;
      case 42:
        v57 = 3LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
        {
          v58 = *(_QWORD **)(v6 - 8);
          v6 = (__int64)&v58[v57];
        }
        else
        {
          v58 = (_QWORD *)(v6 - v57 * 8);
        }
        if ( (_QWORD *)v6 == v58 )
          return 1;
        v59 = v58;
        while ( 1 )
        {
          v87 = v59;
          if ( !(unsigned __int8)sub_174A4F0(*v59, a2, a3, a4, v59, v14) )
            break;
          v59 = v87 + 3;
          if ( (_QWORD *)v6 == v87 + 3 )
            return 1;
        }
        return v4;
      case 44:
        if ( !(unsigned __int8)sub_174A4F0(*(_QWORD *)(v6 - 48), a2, a3, a4, v13, v14) )
          return v4;
        v6 = *(_QWORD *)(v6 - 24);
        goto LABEL_16;
      default:
        return v4;
    }
  }
}
