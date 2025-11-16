// Function: sub_22486A0
// Address: 0x22486a0
//
__int64 __fastcall sub_22486A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8,
        wchar_t *s)
{
  size_t v9; // r14
  _QWORD *v10; // rbp
  __int64 v11; // r13
  _QWORD *v12; // rdi
  size_t v13; // r12
  int v14; // eax
  char v15; // bp
  char v16; // r15
  char v17; // al
  char v18; // dl
  size_t v19; // r15
  wchar_t *v20; // rbp
  size_t v21; // rbp
  char v22; // al
  int v23; // eax
  wchar_t v24; // ebp
  int v25; // eax
  _DWORD *v27; // rax
  int v28; // eax
  int *v29; // rax
  int v30; // ecx
  int *v31; // rax
  int v32; // edx
  int v33; // edx
  _QWORD *v34; // rax
  int v35; // edx
  _QWORD *v36; // rax
  int v37; // edx
  int v38; // edx
  unsigned int v39; // eax
  int v40; // edx
  int v41; // r15d
  int v42; // r14d
  unsigned int v43; // edx
  int v44; // edx
  int v45; // edx
  int v46; // edx
  int v47; // edx
  int v48; // edx
  int v49; // edx
  int v50; // edx
  _QWORD *v51; // rax
  int v52; // edx
  _QWORD *v53; // rax
  int v54; // edx
  int v55; // edx
  unsigned int v56; // eax
  int v57; // edx
  int v58; // edx
  unsigned int v59; // eax
  unsigned int v60; // eax
  int v61; // edx
  int v62; // edx
  int v63; // edx
  int v64; // eax
  __int64 v67; // [rsp+18h] [rbp-240h]
  char v68; // [rsp+20h] [rbp-238h]
  int v69[2]; // [rsp+190h] [rbp-C8h] BYREF
  int v70[2]; // [rsp+198h] [rbp-C0h]
  int v71[2]; // [rsp+1A0h] [rbp-B8h] BYREF
  int v72[2]; // [rsp+1A8h] [rbp-B0h]
  int v73; // [rsp+1B8h] [rbp-A0h] BYREF
  int v74; // [rsp+1BCh] [rbp-9Ch] BYREF
  wchar_t v75[2]; // [rsp+1C0h] [rbp-98h] BYREF
  __int64 v76; // [rsp+1C8h] [rbp-90h]
  __int64 v77; // [rsp+1D0h] [rbp-88h]
  __int64 v78; // [rsp+1D8h] [rbp-80h]
  __int64 v79; // [rsp+1E0h] [rbp-78h]
  __int64 v80; // [rsp+1E8h] [rbp-70h]
  __int64 v81; // [rsp+1F0h] [rbp-68h]
  __int64 v82; // [rsp+1F8h] [rbp-60h]
  __int64 v83; // [rsp+200h] [rbp-58h]
  __int64 v84; // [rsp+208h] [rbp-50h]
  __int64 v85; // [rsp+210h] [rbp-48h]
  __int64 v86; // [rsp+218h] [rbp-40h]

  v9 = 0;
  v10 = (_QWORD *)(a6 + 208);
  *(_QWORD *)v71 = a2;
  *(_QWORD *)v72 = a3;
  *(_QWORD *)v69 = a4;
  *(_QWORD *)v70 = a5;
  v67 = sub_2244AF0((_QWORD *)(a6 + 208), a2);
  v11 = sub_2243120(v10, a2);
  v12 = (_QWORD *)a2;
  v73 = 0;
  v13 = wcslen(s);
  v14 = v72[0];
  while ( 1 )
  {
    v15 = v14 == -1;
    v16 = v15 & (v12 != 0);
    if ( v16 )
    {
      v27 = (_DWORD *)v12[2];
      v28 = (unsigned __int64)v27 >= v12[3] ? (*(__int64 (__fastcall **)(_QWORD *))(*v12 + 72LL))(v12) : *v27;
      v15 = 0;
      if ( v28 == -1 )
      {
        *(_QWORD *)v71 = 0;
        v15 = v16;
      }
    }
    v17 = v70[0] == -1;
    v18 = v17 & (*(_QWORD *)v69 != 0);
    if ( v18 )
    {
      v29 = *(int **)(*(_QWORD *)v69 + 16LL);
      if ( (unsigned __int64)v29 >= *(_QWORD *)(*(_QWORD *)v69 + 24LL) )
      {
        v68 = v18;
        v64 = (*(__int64 (**)(void))(**(_QWORD **)v69 + 72LL))();
        v18 = v68;
        v30 = v64;
      }
      else
      {
        v30 = *v29;
      }
      v17 = 0;
      if ( v30 == -1 )
      {
        *(_QWORD *)v69 = 0;
        v17 = v18;
      }
    }
    if ( v15 == v17 || v9 >= v13 )
      break;
    if ( v73 )
      goto LABEL_22;
    v19 = v9;
    v20 = &s[v9];
    if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v11 + 96LL))(v11, (unsigned int)*v20) == 37 )
    {
      v21 = v9 + 1;
      v22 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v11 + 96LL))(
              v11,
              (unsigned int)s[v19 + 1],
              0);
      v74 = 0;
      if ( v22 == 69 || v22 == 79 )
      {
        v21 = v9 + 2;
        v23 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v11 + 96LL))(
                v11,
                (unsigned int)s[v19 + 2],
                0)
            - 65;
        if ( (unsigned __int8)v23 <= 0x38u )
          goto LABEL_11;
LABEL_18:
        v73 |= 4u;
LABEL_19:
        v14 = v72[0];
      }
      else
      {
        LOBYTE(v23) = v22 - 65;
LABEL_11:
        switch ( (char)v23 )
        {
          case 0:
            v51 = *(_QWORD **)(v67 + 16);
            *(_QWORD *)v75 = v51[11];
            v76 = v51[12];
            v77 = v51[13];
            v78 = v51[14];
            v79 = v51[15];
            v80 = v51[16];
            v81 = v51[17];
            *(_QWORD *)v71 = sub_2247960(
                               a1,
                               *(_QWORD **)v71,
                               *(__int64 *)v72,
                               *(__int64 **)v69,
                               *(__int64 *)v70,
                               &v74,
                               (__int64)v75,
                               7,
                               a6,
                               &v73);
            v14 = v52;
            v72[0] = v52;
            if ( !v73 )
              goto LABEL_46;
            break;
          case 1:
            v53 = *(_QWORD **)(v67 + 16);
            *(_QWORD *)v75 = v53[25];
            v76 = v53[26];
            v77 = v53[27];
            v78 = v53[28];
            v79 = v53[29];
            v80 = v53[30];
            v81 = v53[31];
            v82 = v53[32];
            v83 = v53[33];
            v84 = v53[34];
            v85 = v53[35];
            v86 = v53[36];
            *(_QWORD *)v71 = sub_2247960(
                               a1,
                               *(_QWORD **)v71,
                               *(__int64 *)v72,
                               *(__int64 **)v69,
                               *(__int64 *)v70,
                               &v74,
                               (__int64)v75,
                               12,
                               a6,
                               &v73);
            v14 = v54;
            v72[0] = v54;
            if ( !v73 )
              goto LABEL_44;
            break;
          case 2:
          case 24:
          case 56:
            *(_QWORD *)v71 = sub_2243170(
                               a1,
                               *(__int64 *)v71,
                               v72[0],
                               *(_QWORD **)v69,
                               v70[0],
                               &v74,
                               0,
                               9999,
                               4u,
                               a6,
                               &v73);
            v14 = v32;
            v72[0] = v32;
            if ( !v73 )
            {
              v33 = v74 - 1900;
              if ( v74 < 0 )
                v33 = v74 + 100;
              a8[5] = v33;
            }
            break;
          case 3:
            (*(void (__fastcall **)(__int64, char *, const char *, wchar_t *))(*(_QWORD *)v11 + 88LL))(
              v11,
              &aHM[-9],
              "%H:%M",
              v75);
            *(_QWORD *)v71 = sub_22486A0(a1, v71[0], v72[0], v69[0], v70[0], a6, (__int64)&v73, (__int64)a8, v75);
            v14 = v55;
            v72[0] = v55;
            break;
          case 7:
            *(_QWORD *)v71 = sub_2243170(
                               a1,
                               *(__int64 *)v71,
                               v72[0],
                               *(_QWORD **)v69,
                               v70[0],
                               &v74,
                               0,
                               23,
                               2u,
                               a6,
                               &v73);
            v14 = v47;
            v72[0] = v47;
            if ( !v73 )
              goto LABEL_58;
            break;
          case 8:
            *(_QWORD *)v71 = sub_2243170(
                               a1,
                               *(__int64 *)v71,
                               v72[0],
                               *(_QWORD **)v69,
                               v70[0],
                               &v74,
                               1,
                               12,
                               2u,
                               a6,
                               &v73);
            v14 = v48;
            v72[0] = v48;
            if ( !v73 )
LABEL_58:
              a8[2] = v74;
            break;
          case 12:
            *(_QWORD *)v71 = sub_2243170(
                               a1,
                               *(__int64 *)v71,
                               v72[0],
                               *(_QWORD **)v69,
                               v70[0],
                               &v74,
                               0,
                               59,
                               2u,
                               a6,
                               &v73);
            v14 = v49;
            v72[0] = v49;
            if ( !v73 )
              a8[1] = v74;
            break;
          case 17:
            (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, wchar_t *))(*(_QWORD *)v11 + 88LL))(
              v11,
              &byte_4360B49[-6],
              byte_4360B49,
              v75);
            *(_QWORD *)v71 = sub_22486A0(a1, v71[0], v72[0], v69[0], v70[0], a6, (__int64)&v73, (__int64)a8, v75);
            v14 = v50;
            v72[0] = v50;
            break;
          case 18:
            *(_QWORD *)v71 = sub_2243170(
                               a1,
                               *(__int64 *)v71,
                               v72[0],
                               *(_QWORD **)v69,
                               v70[0],
                               &v74,
                               0,
                               60,
                               2u,
                               a6,
                               &v73);
            v14 = v62;
            v72[0] = v62;
            if ( !v73 )
              *a8 = v74;
            break;
          case 19:
            (*(void (**)(__int64, char *, const char *, ...))(*(_QWORD *)v11 + 88LL))(v11, &a9lu[-9], "%.9lu", v75);
            *(_QWORD *)v71 = sub_22486A0(a1, v71[0], v72[0], v69[0], v70[0], a6, (__int64)&v73, (__int64)a8, v75);
            v14 = v63;
            v72[0] = v63;
            break;
          case 23:
            *(_QWORD *)v71 = sub_22486A0(
                               a1,
                               v71[0],
                               v72[0],
                               v69[0],
                               v70[0],
                               a6,
                               (__int64)&v73,
                               (__int64)a8,
                               *(wchar_t **)(*(_QWORD *)(v67 + 16) + 32LL));
            v14 = v38;
            v72[0] = v38;
            break;
          case 25:
            v39 = sub_2247910((__int64)v71);
            if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v11 + 16LL))(v11, 256, v39) )
              goto LABEL_18;
            *(_QWORD *)v71 = sub_2247960(
                               a1,
                               *(_QWORD **)v71,
                               *(__int64 *)v72,
                               *(__int64 **)v69,
                               *(__int64 *)v70,
                               v75,
                               (__int64)&off_4CDFAE0,
                               14,
                               a6,
                               &v73);
            v72[0] = v40;
            if ( !sub_2247850((__int64)v71, (__int64)v69) && !(v75[0] | v73) )
            {
              v41 = sub_2247910((__int64)v71);
              if ( v41 == (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 80LL))(v11, 45)
                || (v42 = sub_2247910((__int64)v71),
                    v42 == (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 80LL))(v11, 43)) )
              {
                *(_QWORD *)v71 = sub_2243170(
                                   a1,
                                   *(__int64 *)v71,
                                   v72[0],
                                   *(_QWORD **)v69,
                                   v70[0],
                                   v75,
                                   0,
                                   23,
                                   2u,
                                   a6,
                                   &v73);
                v72[0] = v43;
                *(_QWORD *)v71 = sub_2243170(
                                   a1,
                                   *(__int64 *)v71,
                                   v43,
                                   *(_QWORD **)v69,
                                   v70[0],
                                   v75,
                                   0,
                                   59,
                                   2u,
                                   a6,
                                   &v73);
                v72[0] = v44;
              }
            }
            goto LABEL_19;
          case 32:
            v36 = *(_QWORD **)(v67 + 16);
            *(_QWORD *)v75 = v36[18];
            v76 = v36[19];
            v77 = v36[20];
            v78 = v36[21];
            v79 = v36[22];
            v80 = v36[23];
            v81 = v36[24];
            *(_QWORD *)v71 = sub_2247960(
                               a1,
                               *(_QWORD **)v71,
                               *(__int64 *)v72,
                               *(__int64 **)v69,
                               *(__int64 *)v70,
                               &v74,
                               (__int64)v75,
                               7,
                               a6,
                               &v73);
            v14 = v37;
            v72[0] = v37;
            if ( !v73 )
LABEL_46:
              a8[6] = v74;
            break;
          case 33:
          case 39:
            v34 = *(_QWORD **)(v67 + 16);
            *(_QWORD *)v75 = v34[37];
            v76 = v34[38];
            v77 = v34[39];
            v78 = v34[40];
            v79 = v34[41];
            v80 = v34[42];
            v81 = v34[43];
            v82 = v34[44];
            v83 = v34[45];
            v84 = v34[46];
            v85 = v34[47];
            v86 = v34[48];
            *(_QWORD *)v71 = sub_2247960(
                               a1,
                               *(_QWORD **)v71,
                               *(__int64 *)v72,
                               *(__int64 **)v69,
                               *(__int64 *)v70,
                               &v74,
                               (__int64)v75,
                               12,
                               a6,
                               &v73);
            v14 = v35;
            v72[0] = v35;
            if ( !v73 )
LABEL_44:
              a8[4] = v74;
            break;
          case 34:
            *(_QWORD *)v71 = sub_22486A0(
                               a1,
                               v71[0],
                               v72[0],
                               v69[0],
                               v70[0],
                               a6,
                               (__int64)&v73,
                               (__int64)a8,
                               *(wchar_t **)(*(_QWORD *)(v67 + 16) + 48LL));
            v14 = v45;
            v72[0] = v45;
            break;
          case 35:
            *(_QWORD *)v71 = sub_2243170(
                               a1,
                               *(__int64 *)v71,
                               v72[0],
                               *(_QWORD **)v69,
                               v70[0],
                               &v74,
                               1,
                               31,
                               2u,
                               a6,
                               &v73);
            v14 = v46;
            v72[0] = v46;
            if ( !v73 )
              goto LABEL_56;
            break;
          case 36:
            v56 = sub_2247910((__int64)v71);
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v11 + 16LL))(v11, 0x2000, v56) )
            {
              sub_2240940(*(_QWORD **)v71);
              v72[0] = -1;
              *(_QWORD *)v71 = sub_2243170(
                                 a1,
                                 *(__int64 *)v71,
                                 0xFFFFFFFF,
                                 *(_QWORD **)v69,
                                 v70[0],
                                 &v74,
                                 1,
                                 9,
                                 1u,
                                 a6,
                                 &v73);
            }
            else
            {
              *(_QWORD *)v71 = sub_2243170(
                                 a1,
                                 *(__int64 *)v71,
                                 v72[0],
                                 *(_QWORD **)v69,
                                 v70[0],
                                 &v74,
                                 10,
                                 31,
                                 2u,
                                 a6,
                                 &v73);
            }
            v72[0] = v57;
            v14 = v57;
            if ( !v73 )
LABEL_56:
              a8[3] = v74;
            break;
          case 44:
            *(_QWORD *)v71 = sub_2243170(
                               a1,
                               *(__int64 *)v71,
                               v72[0],
                               *(_QWORD **)v69,
                               v70[0],
                               &v74,
                               1,
                               12,
                               2u,
                               a6,
                               &v73);
            v14 = v58;
            v72[0] = v58;
            if ( !v73 )
              a8[4] = v74 - 1;
            break;
          case 45:
            v59 = sub_2247910((__int64)v71);
            if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v11 + 96LL))(v11, v59, 0) != 10 )
              goto LABEL_18;
            goto LABEL_76;
          case 51:
            v60 = sub_2247910((__int64)v71);
            if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v11 + 96LL))(v11, v60, 0) != 9 )
              goto LABEL_18;
LABEL_76:
            sub_2240940(*(_QWORD **)v71);
            v14 = -1;
            v72[0] = -1;
            break;
          case 55:
            *(_QWORD *)v71 = sub_22486A0(
                               a1,
                               v71[0],
                               v72[0],
                               v69[0],
                               v70[0],
                               a6,
                               (__int64)&v73,
                               (__int64)a8,
                               *(wchar_t **)(*(_QWORD *)(v67 + 16) + 16LL));
            v14 = v61;
            v72[0] = v61;
            break;
          default:
            goto LABEL_18;
        }
      }
      v12 = *(_QWORD **)v71;
      goto LABEL_16;
    }
    v12 = *(_QWORD **)v71;
    v24 = *v20;
    v25 = v72[0];
    if ( *(_QWORD *)v71 && v72[0] == -1 )
    {
      v31 = *(int **)(*(_QWORD *)v71 + 16LL);
      if ( (unsigned __int64)v31 >= *(_QWORD *)(*(_QWORD *)v71 + 24LL) )
        v25 = (*(__int64 (**)(void))(**(_QWORD **)v71 + 72LL))();
      else
        v25 = *v31;
      if ( v25 == -1 )
      {
        *(_QWORD *)v71 = 0;
        v12 = 0;
      }
      else
      {
        v12 = *(_QWORD **)v71;
      }
    }
    if ( v24 == v25 )
    {
      sub_2240940(v12);
      v21 = v9;
      v14 = -1;
      v72[0] = -1;
      v12 = *(_QWORD **)v71;
    }
    else
    {
      v73 |= 4u;
      v14 = v72[0];
      v21 = v9;
    }
LABEL_16:
    v9 = v21 + 1;
  }
  if ( v9 != v13 || v73 )
LABEL_22:
    *a7 |= 4u;
  return *(_QWORD *)v71;
}
