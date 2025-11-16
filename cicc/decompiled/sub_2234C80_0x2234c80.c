// Function: sub_2234C80
// Address: 0x2234c80
//
__int64 __fastcall sub_2234C80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8,
        char *s)
{
  size_t v9; // rbx
  _QWORD *v10; // rdi
  __int64 v11; // rbp
  size_t v12; // rax
  int *v13; // rsi
  size_t v14; // r15
  bool v15; // al
  char *v16; // rcx
  __int64 v17; // r8
  char v18; // dl
  char v19; // al
  __int64 (__fastcall *v20)(__int64, unsigned int); // r9
  __int64 v21; // rdi
  __int64 v22; // rdx
  int v23; // eax
  __int64 v25; // rcx
  size_t v26; // r8
  char v27; // al
  char v28; // si
  __int64 (__fastcall *v29)(__int64, unsigned int); // r9
  _BYTE *v30; // rax
  int v31; // edx
  int v32; // eax
  _QWORD *v33; // rax
  int v34; // edx
  int v35; // edx
  int v36; // edx
  int v37; // edx
  int v38; // edx
  _QWORD *v39; // rax
  int v40; // edx
  int v41; // edx
  int v42; // edx
  char v43; // al
  char v44; // al
  int v45; // edx
  int v46; // edx
  int v47; // edx
  int v48; // edx
  int v49; // edx
  char v50; // al
  char v51; // al
  int v52; // edx
  int v53; // edx
  _QWORD *v54; // rax
  int v55; // edx
  _QWORD *v56; // rax
  int v57; // edx
  int v58; // edx
  int v59; // edx
  int v60; // edx
  char v61; // al
  size_t v64; // [rsp+18h] [rbp-250h]
  __int64 v65; // [rsp+20h] [rbp-248h]
  unsigned __int8 v66; // [rsp+28h] [rbp-240h]
  __int64 v67; // [rsp+30h] [rbp-238h]
  __int64 v68; // [rsp+30h] [rbp-238h]
  char v69; // [rsp+3Ch] [rbp-22Ch]
  char v70; // [rsp+3Ch] [rbp-22Ch]
  int v71[2]; // [rsp+1A0h] [rbp-C8h] BYREF
  int v72[2]; // [rsp+1A8h] [rbp-C0h]
  int v73[2]; // [rsp+1B0h] [rbp-B8h] BYREF
  int v74[2]; // [rsp+1B8h] [rbp-B0h]
  int v75; // [rsp+1C8h] [rbp-A0h] BYREF
  int v76; // [rsp+1CCh] [rbp-9Ch] BYREF
  __int64 v77; // [rsp+1D0h] [rbp-98h] BYREF
  __int64 v78; // [rsp+1D8h] [rbp-90h]
  __int64 v79; // [rsp+1E0h] [rbp-88h]
  __int64 v80; // [rsp+1E8h] [rbp-80h]
  __int64 v81; // [rsp+1F0h] [rbp-78h]
  __int64 v82; // [rsp+1F8h] [rbp-70h]
  __int64 v83; // [rsp+200h] [rbp-68h]
  __int64 v84; // [rsp+208h] [rbp-60h]
  __int64 v85; // [rsp+210h] [rbp-58h]
  __int64 v86; // [rsp+218h] [rbp-50h]
  __int64 v87; // [rsp+220h] [rbp-48h]
  __int64 v88; // [rsp+228h] [rbp-40h]

  v9 = 0;
  *(_QWORD *)v73 = a2;
  *(_QWORD *)v74 = a3;
  *(_QWORD *)v71 = a4;
  *(_QWORD *)v72 = a5;
  v10 = (_QWORD *)(a6 + 208);
  v65 = sub_22311C0((_QWORD *)(a6 + 208), a2);
  v11 = sub_222F790(v10, a2);
  v12 = strlen(s);
  v13 = v71;
  v75 = 0;
  v14 = v12;
  v64 = v12;
  v15 = sub_2233E50((__int64)v73, (__int64)v71);
  if ( v14 )
  {
    do
    {
      if ( v15 )
        break;
      if ( v75 )
        goto LABEL_17;
      v16 = &s[v9];
      v17 = (unsigned __int8)s[v9];
      v18 = *(_BYTE *)(v11 + v17 + 313);
      v19 = s[v9];
      if ( !v18 )
      {
        v20 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v11 + 64LL);
        if ( v20 != sub_2216C50 )
        {
          v67 = (unsigned __int8)s[v9];
          v13 = (int *)(unsigned int)(char)v17;
          v19 = ((__int64 (__fastcall *)(__int64, int *, _QWORD))v20)(v11, v13, 0);
          v17 = v67;
          v16 = &s[v9];
        }
        if ( !v19 )
          goto LABEL_10;
        *(_BYTE *)(v11 + v17 + 313) = v19;
        v18 = v19;
      }
      if ( v18 == 37 )
      {
        v25 = (unsigned __int8)s[v9 + 1];
        v26 = v9 + 1;
        v27 = *(_BYTE *)(v11 + v25 + 313);
        v28 = s[v9 + 1];
        if ( v27 )
          goto LABEL_24;
        v29 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v11 + 64LL);
        if ( v29 != sub_2216C50 )
        {
          v68 = (unsigned __int8)s[v9 + 1];
          v61 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v29)(v11, (unsigned int)(char)v25, 0);
          v25 = v68;
          v26 = v9 + 1;
          v28 = v61;
        }
        if ( v28 )
        {
          *(_BYTE *)(v11 + v25 + 313) = v28;
          v27 = v28;
LABEL_24:
          v76 = 0;
          if ( v27 == 69 || v27 == 79 )
          {
            v9 += 2LL;
            v27 = sub_222F680(v11, s[v9], 0);
          }
          else
          {
            v9 = v26;
          }
          switch ( v27 )
          {
            case 'A':
              v56 = *(_QWORD **)(v65 + 16);
              v77 = v56[11];
              v78 = v56[12];
              v79 = v56[13];
              v80 = v56[14];
              v81 = v56[15];
              v82 = v56[16];
              v83 = v56[17];
              *(_QWORD *)v73 = sub_2233F50(
                                 a1,
                                 *(__int64 *)v73,
                                 *(__int64 *)v74,
                                 *(__int64 **)v71,
                                 *(__int64 *)v72,
                                 &v76,
                                 (__int64)&v77,
                                 7,
                                 a6,
                                 &v75);
              v74[0] = v57;
              if ( !v75 )
                goto LABEL_76;
              goto LABEL_14;
            case 'B':
              v39 = *(_QWORD **)(v65 + 16);
              v77 = v39[25];
              v78 = v39[26];
              v79 = v39[27];
              v80 = v39[28];
              v81 = v39[29];
              v82 = v39[30];
              v83 = v39[31];
              v84 = v39[32];
              v85 = v39[33];
              v86 = v39[34];
              v87 = v39[35];
              v88 = v39[36];
              *(_QWORD *)v73 = sub_2233F50(
                                 a1,
                                 *(__int64 *)v73,
                                 *(__int64 *)v74,
                                 *(__int64 **)v71,
                                 *(__int64 *)v72,
                                 &v76,
                                 (__int64)&v77,
                                 12,
                                 a6,
                                 &v75);
              v74[0] = v40;
              if ( !v75 )
                goto LABEL_41;
              goto LABEL_14;
            case 'C':
            case 'Y':
            case 'y':
              *(_QWORD *)v73 = sub_222F7E0(
                                 a1,
                                 *(__int64 *)v73,
                                 v74[0],
                                 *(_QWORD **)v71,
                                 v72[0],
                                 &v76,
                                 0,
                                 9999,
                                 4u,
                                 a6,
                                 &v75);
              v74[0] = v31;
              if ( !v75 )
              {
                v32 = v76 - 1900;
                if ( v76 < 0 )
                  v32 = v76 + 100;
                a8[5] = v32;
              }
              goto LABEL_14;
            case 'D':
              sub_222F5F0((_BYTE *)v11, &aHM[-9], "%H:%M", &v77);
              *(_QWORD *)v73 = sub_2234C80(
                                 a1,
                                 v73[0],
                                 v74[0],
                                 v71[0],
                                 v72[0],
                                 a6,
                                 (__int64)&v75,
                                 (__int64)a8,
                                 (char *)&v77);
              v74[0] = v58;
              goto LABEL_14;
            case 'H':
              *(_QWORD *)v73 = sub_222F7E0(
                                 a1,
                                 *(__int64 *)v73,
                                 v74[0],
                                 *(_QWORD **)v71,
                                 v72[0],
                                 &v76,
                                 0,
                                 23,
                                 2u,
                                 a6,
                                 &v75);
              v74[0] = v59;
              if ( !v75 )
                goto LABEL_43;
              goto LABEL_14;
            case 'I':
              *(_QWORD *)v73 = sub_222F7E0(
                                 a1,
                                 *(__int64 *)v73,
                                 v74[0],
                                 *(_QWORD **)v71,
                                 v72[0],
                                 &v76,
                                 1,
                                 12,
                                 2u,
                                 a6,
                                 &v75);
              v74[0] = v35;
              if ( !v75 )
LABEL_43:
                a8[2] = v76;
              goto LABEL_14;
            case 'M':
              *(_QWORD *)v73 = sub_222F7E0(
                                 a1,
                                 *(__int64 *)v73,
                                 v74[0],
                                 *(_QWORD **)v71,
                                 v72[0],
                                 &v76,
                                 0,
                                 59,
                                 2u,
                                 a6,
                                 &v75);
              v74[0] = v36;
              if ( !v75 )
                a8[1] = v76;
              goto LABEL_14;
            case 'R':
              sub_222F5F0((_BYTE *)v11, &byte_4360B49[-6], byte_4360B49, &v77);
              *(_QWORD *)v73 = sub_2234C80(
                                 a1,
                                 v73[0],
                                 v74[0],
                                 v71[0],
                                 v72[0],
                                 a6,
                                 (__int64)&v75,
                                 (__int64)a8,
                                 (char *)&v77);
              v74[0] = v37;
              goto LABEL_14;
            case 'S':
              *(_QWORD *)v73 = sub_222F7E0(
                                 a1,
                                 *(__int64 *)v73,
                                 v74[0],
                                 *(_QWORD **)v71,
                                 v72[0],
                                 &v76,
                                 0,
                                 60,
                                 2u,
                                 a6,
                                 &v75);
              v74[0] = v38;
              if ( !v75 )
                *a8 = v76;
              goto LABEL_14;
            case 'T':
              sub_222F5F0((_BYTE *)v11, &a9lu[-9], "%.9lu", &v77);
              *(_QWORD *)v73 = sub_2234C80(
                                 a1,
                                 v73[0],
                                 v74[0],
                                 v71[0],
                                 v72[0],
                                 a6,
                                 (__int64)&v75,
                                 (__int64)a8,
                                 (char *)&v77);
              v74[0] = v47;
              goto LABEL_14;
            case 'X':
              *(_QWORD *)v73 = sub_2234C80(
                                 a1,
                                 v73[0],
                                 v74[0],
                                 v71[0],
                                 v72[0],
                                 a6,
                                 (__int64)&v75,
                                 (__int64)a8,
                                 *(char **)(*(_QWORD *)(v65 + 16) + 32LL));
              v74[0] = v48;
              goto LABEL_14;
            case 'Z':
              if ( (*(_BYTE *)(*(_QWORD *)(v11 + 48) + 2LL * (unsigned __int8)sub_2233F00((__int64)v73) + 1) & 1) == 0 )
                goto LABEL_13;
              *(_QWORD *)v73 = sub_2233F50(
                                 a1,
                                 *(__int64 *)v73,
                                 *(__int64 *)v74,
                                 *(__int64 **)v71,
                                 *(__int64 *)v72,
                                 &v77,
                                 (__int64)off_4CDFB60,
                                 14,
                                 a6,
                                 &v75);
              v74[0] = v49;
              if ( !sub_2233E50((__int64)v73, (__int64)v71) && !((unsigned int)v77 | v75) )
              {
                v69 = sub_2233F00((__int64)v73);
                v50 = *(_BYTE *)(v11 + 56) ? *(_BYTE *)(v11 + 102) : sub_222EC20(v11, 0x2Du);
                if ( v69 == v50
                  || ((v70 = sub_2233F00((__int64)v73), !*(_BYTE *)(v11 + 56))
                    ? (v51 = sub_222EC20(v11, 0x2Bu))
                    : (v51 = *(_BYTE *)(v11 + 100)),
                      v70 == v51) )
                {
                  *(_QWORD *)v73 = sub_222F7E0(
                                     a1,
                                     *(__int64 *)v73,
                                     v74[0],
                                     *(_QWORD **)v71,
                                     v72[0],
                                     (int *)&v77,
                                     0,
                                     23,
                                     2u,
                                     a6,
                                     &v75);
                  v74[0] = v52;
                  *(_QWORD *)v73 = sub_222F7E0(
                                     a1,
                                     *(__int64 *)v73,
                                     v52,
                                     *(_QWORD **)v71,
                                     v72[0],
                                     (int *)&v77,
                                     0,
                                     59,
                                     2u,
                                     a6,
                                     &v75);
                  v74[0] = v53;
                }
              }
              goto LABEL_14;
            case 'a':
              v54 = *(_QWORD **)(v65 + 16);
              v77 = v54[18];
              v78 = v54[19];
              v79 = v54[20];
              v80 = v54[21];
              v81 = v54[22];
              v82 = v54[23];
              v83 = v54[24];
              *(_QWORD *)v73 = sub_2233F50(
                                 a1,
                                 *(__int64 *)v73,
                                 *(__int64 *)v74,
                                 *(__int64 **)v71,
                                 *(__int64 *)v72,
                                 &v76,
                                 (__int64)&v77,
                                 7,
                                 a6,
                                 &v75);
              v74[0] = v55;
              if ( !v75 )
LABEL_76:
                a8[6] = v76;
              goto LABEL_14;
            case 'b':
            case 'h':
              v33 = *(_QWORD **)(v65 + 16);
              v77 = v33[37];
              v78 = v33[38];
              v79 = v33[39];
              v80 = v33[40];
              v81 = v33[41];
              v82 = v33[42];
              v83 = v33[43];
              v84 = v33[44];
              v85 = v33[45];
              v86 = v33[46];
              v87 = v33[47];
              v88 = v33[48];
              *(_QWORD *)v73 = sub_2233F50(
                                 a1,
                                 *(__int64 *)v73,
                                 *(__int64 *)v74,
                                 *(__int64 **)v71,
                                 *(__int64 *)v72,
                                 &v76,
                                 (__int64)&v77,
                                 12,
                                 a6,
                                 &v75);
              v74[0] = v34;
              if ( !v75 )
LABEL_41:
                a8[4] = v76;
              goto LABEL_14;
            case 'c':
              *(_QWORD *)v73 = sub_2234C80(
                                 a1,
                                 v73[0],
                                 v74[0],
                                 v71[0],
                                 v72[0],
                                 a6,
                                 (__int64)&v75,
                                 (__int64)a8,
                                 *(char **)(*(_QWORD *)(v65 + 16) + 48LL));
              v74[0] = v42;
              goto LABEL_14;
            case 'd':
              *(_QWORD *)v73 = sub_222F7E0(
                                 a1,
                                 *(__int64 *)v73,
                                 v74[0],
                                 *(_QWORD **)v71,
                                 v72[0],
                                 &v76,
                                 1,
                                 31,
                                 2u,
                                 a6,
                                 &v75);
              v74[0] = v46;
              if ( !v75 )
                goto LABEL_54;
              goto LABEL_14;
            case 'e':
              if ( (*(_BYTE *)(*(_QWORD *)(v11 + 48) + 2LL * (unsigned __int8)sub_2233F00((__int64)v73) + 1) & 0x20) != 0 )
              {
                sub_22408B0(*(_QWORD *)v73);
                v74[0] = -1;
                *(_QWORD *)v73 = sub_222F7E0(a1, *(__int64 *)v73, -1, *(_QWORD **)v71, v72[0], &v76, 1, 9, 1u, a6, &v75);
              }
              else
              {
                *(_QWORD *)v73 = sub_222F7E0(
                                   a1,
                                   *(__int64 *)v73,
                                   v74[0],
                                   *(_QWORD **)v71,
                                   v72[0],
                                   &v76,
                                   10,
                                   31,
                                   2u,
                                   a6,
                                   &v75);
              }
              v74[0] = v41;
              if ( !v75 )
LABEL_54:
                a8[3] = v76;
              goto LABEL_14;
            case 'm':
              *(_QWORD *)v73 = sub_222F7E0(
                                 a1,
                                 *(__int64 *)v73,
                                 v74[0],
                                 *(_QWORD **)v71,
                                 v72[0],
                                 &v76,
                                 1,
                                 12,
                                 2u,
                                 a6,
                                 &v75);
              v74[0] = v60;
              if ( !v75 )
                a8[4] = v76 - 1;
              goto LABEL_14;
            case 'n':
              v43 = sub_2233F00((__int64)v73);
              if ( (unsigned __int8)sub_222F680(v11, v43, 0) != 10 )
                goto LABEL_13;
              goto LABEL_57;
            case 't':
              v44 = sub_2233F00((__int64)v73);
              if ( (unsigned __int8)sub_222F680(v11, v44, 0) != 9 )
                goto LABEL_13;
LABEL_57:
              sub_22408B0(*(_QWORD *)v73);
              v74[0] = -1;
              break;
            case 'x':
              *(_QWORD *)v73 = sub_2234C80(
                                 a1,
                                 v73[0],
                                 v74[0],
                                 v71[0],
                                 v72[0],
                                 a6,
                                 (__int64)&v75,
                                 (__int64)a8,
                                 *(char **)(*(_QWORD *)(v65 + 16) + 16LL));
              v74[0] = v45;
              break;
            default:
              goto LABEL_13;
          }
          goto LABEL_14;
        }
        v9 = v26;
        goto LABEL_13;
      }
LABEL_10:
      v21 = *(_QWORD *)v73;
      v22 = (unsigned __int8)*v16;
      LOBYTE(v23) = v74[0];
      if ( *(_QWORD *)v73 && v74[0] == -1 )
      {
        v30 = *(_BYTE **)(*(_QWORD *)v73 + 16LL);
        if ( (unsigned __int64)v30 < *(_QWORD *)(*(_QWORD *)v73 + 24LL) )
        {
          if ( (_BYTE)v22 == *v30 )
            goto LABEL_29;
          goto LABEL_13;
        }
        v66 = *v16;
        v23 = (*(__int64 (__fastcall **)(_QWORD, int *, __int64, char *, __int64))(**(_QWORD **)v73 + 72LL))(
                *(_QWORD *)v73,
                v13,
                v22,
                v16,
                v17);
        v22 = v66;
        if ( v23 == -1 )
          *(_QWORD *)v73 = 0;
      }
      if ( (_BYTE)v22 == (_BYTE)v23 )
      {
        v21 = *(_QWORD *)v73;
        v30 = *(_BYTE **)(*(_QWORD *)v73 + 16LL);
        if ( (unsigned __int64)v30 >= *(_QWORD *)(*(_QWORD *)v73 + 24LL) )
          (*(void (__fastcall **)(_QWORD, int *, __int64, char *, __int64))(**(_QWORD **)v73 + 80LL))(
            *(_QWORD *)v73,
            v13,
            v22,
            v16,
            v17);
        else
LABEL_29:
          *(_QWORD *)(v21 + 16) = v30 + 1;
        v74[0] = -1;
        goto LABEL_14;
      }
LABEL_13:
      v75 |= 4u;
LABEL_14:
      ++v9;
      v13 = v71;
      v15 = sub_2233E50((__int64)v73, (__int64)v71);
    }
    while ( v9 < v14 );
  }
  if ( v75 || v9 != v64 )
LABEL_17:
    *a7 |= 4u;
  return *(_QWORD *)v73;
}
