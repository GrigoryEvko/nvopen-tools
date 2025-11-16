// Function: sub_1106750
// Address: 0x1106750
//
__int64 __fastcall sub_1106750(__int64 a1, unsigned __int8 *a2, __int64 a3, unsigned __int8 a4)
{
  unsigned __int8 *v6; // r12
  int v8; // eax
  int v10; // r11d
  unsigned int v11; // r15d
  unsigned __int8 *v12; // rsi
  __int64 v13; // r13
  unsigned __int8 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int8 *v17; // r11
  __int64 v18; // rbx
  const char *v19; // rsi
  __int64 v20; // r12
  __int64 v21; // rsi
  unsigned __int8 *v22; // rsi
  __int64 v23; // rdi
  __int64 *v24; // rdi
  __int64 *v25; // rsi
  unsigned int v26; // r13d
  unsigned __int8 *v27; // rdx
  __int64 v28; // r15
  unsigned __int8 *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rbx
  unsigned __int8 *v32; // rdx
  unsigned __int8 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  int v40; // r15d
  __int64 v41; // rax
  int v42; // r15d
  __int64 v43; // r10
  __int64 v44; // r15
  __int64 v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rcx
  int v49; // eax
  int v50; // eax
  unsigned int v51; // esi
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rsi
  __int64 *v55; // rdi
  unsigned __int8 *v56; // rdx
  unsigned int v57; // r15d
  __int64 v58; // rax
  __int64 v59; // r13
  unsigned __int8 *v60; // rsi
  __int64 v61; // rbx
  unsigned __int8 *v62; // rdx
  _QWORD *v63; // rax
  __int64 v64; // rax
  __int64 *v65; // rax
  __int64 v66; // rax
  __int64 v67; // rbx
  _QWORD *v68; // rax
  __int64 v69; // [rsp+8h] [rbp-A8h]
  __int64 v70; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v71; // [rsp+20h] [rbp-90h]
  __int64 v72; // [rsp+20h] [rbp-90h]
  int v73; // [rsp+28h] [rbp-88h]
  __int64 v74; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v75; // [rsp+28h] [rbp-88h]
  unsigned int v76; // [rsp+28h] [rbp-88h]
  __int64 v77; // [rsp+28h] [rbp-88h]
  __int64 v78; // [rsp+28h] [rbp-88h]
  __int64 v80; // [rsp+30h] [rbp-80h]
  __int64 v81; // [rsp+30h] [rbp-80h]
  unsigned __int8 *v82; // [rsp+30h] [rbp-80h]
  __int64 v84; // [rsp+30h] [rbp-80h]
  void *v85; // [rsp+30h] [rbp-80h]
  __int64 v86; // [rsp+48h] [rbp-68h]
  const char *v87[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v88; // [rsp+70h] [rbp-40h]

  v6 = a2;
  v8 = *a2;
  if ( (unsigned __int8)v8 <= 0x15u )
    return sub_96F3F0((__int64)a2, a3, a4, *(_QWORD *)(a1 + 88));
  v10 = v8 - 29;
  switch ( *a2 )
  {
    case '*':
    case ',':
    case '.':
    case '0':
    case '3':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
      v11 = a4;
      if ( (a2[7] & 0x40) != 0 )
        v12 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v12 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v73 = v8 - 29;
      v13 = sub_1106750(a1, *(_QWORD *)v12, a3, a4);
      if ( (v6[7] & 0x40) != 0 )
        v14 = (unsigned __int8 *)*((_QWORD *)v6 - 1);
      else
        v14 = &v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
      v15 = sub_1106750(a1, *((_QWORD *)v14 + 4), a3, v11);
      v88 = 257;
      v16 = sub_B504D0(v73, v13, v15, (__int64)v87, 0, 0);
      v17 = (unsigned __int8 *)v16;
      break;
    case 'C':
    case 'D':
    case 'E':
      if ( (a2[7] & 0x40) != 0 )
        v24 = (__int64 *)*((_QWORD *)a2 - 1);
      else
        v24 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v16 = *v24;
      if ( a3 == *(_QWORD *)(*v24 + 8) )
        return v16;
      v88 = 257;
      v16 = sub_B522D0(*v24, a3, v10 == 40, (__int64)v87, 0, 0);
      v17 = (unsigned __int8 *)v16;
      break;
    case 'F':
    case 'G':
      v88 = 257;
      if ( (a2[7] & 0x40) != 0 )
        v25 = (__int64 *)*((_QWORD *)a2 - 1);
      else
        v25 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v16 = sub_B51D30(v10, *v25, a3, (__int64)v87, 0, 0);
      v17 = (unsigned __int8 *)v16;
      break;
    case 'T':
      v40 = *((_DWORD *)a2 + 1);
      v88 = 257;
      v41 = sub_BD2DA0(80);
      v42 = v40 & 0x7FFFFFF;
      v43 = a3;
      v16 = v41;
      if ( v41 )
      {
        v75 = (unsigned __int8 *)v41;
        sub_B44260(v41, a3, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v16 + 72) = v42;
        sub_BD6B50((unsigned __int8 *)v16, v87);
        sub_BD2A10(v16, *(_DWORD *)(v16 + 72), 1);
        v17 = v75;
        v43 = a3;
      }
      else
      {
        v17 = 0;
      }
      if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) != 0 )
      {
        v71 = v17;
        v76 = a4;
        v84 = 8LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
        v44 = 0;
        v46 = v43;
        do
        {
          v47 = sub_1106750(a1, *(_QWORD *)(*((_QWORD *)a2 - 1) + 4 * v44), v46, v76);
          v48 = *(_QWORD *)(*((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18) + v44);
          v49 = *(_DWORD *)(v16 + 4) & 0x7FFFFFF;
          if ( v49 == *(_DWORD *)(v16 + 72) )
          {
            v69 = *(_QWORD *)(*((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18) + v44);
            v70 = v47;
            sub_B48D90(v16);
            v48 = v69;
            v47 = v70;
            v49 = *(_DWORD *)(v16 + 4) & 0x7FFFFFF;
          }
          v50 = (v49 + 1) & 0x7FFFFFF;
          v51 = v50 | *(_DWORD *)(v16 + 4) & 0xF8000000;
          v52 = *(_QWORD *)(v16 - 8) + 32LL * (unsigned int)(v50 - 1);
          *(_DWORD *)(v16 + 4) = v51;
          if ( *(_QWORD *)v52 )
          {
            v53 = *(_QWORD *)(v52 + 8);
            **(_QWORD **)(v52 + 16) = v53;
            if ( v53 )
              *(_QWORD *)(v53 + 16) = *(_QWORD *)(v52 + 16);
          }
          *(_QWORD *)v52 = v47;
          if ( v47 )
          {
            v54 = *(_QWORD *)(v47 + 16);
            *(_QWORD *)(v52 + 8) = v54;
            if ( v54 )
              *(_QWORD *)(v54 + 16) = v52 + 8;
            *(_QWORD *)(v52 + 16) = v47 + 16;
            *(_QWORD *)(v47 + 16) = v52;
          }
          v44 += 8;
          *(_QWORD *)(*(_QWORD *)(v16 - 8)
                    + 32LL * *(unsigned int *)(v16 + 72)
                    + 8LL * ((*(_DWORD *)(v16 + 4) & 0x7FFFFFFu) - 1)) = v48;
        }
        while ( v84 != v44 );
        v17 = v71;
        v6 = a2;
      }
      break;
    case 'U':
      v64 = *((_QWORD *)a2 - 4);
      if ( v64 && !*(_BYTE *)v64 && *(_QWORD *)(v64 + 24) == *((_QWORD *)a2 + 10) && (*(_BYTE *)(v64 + 33) & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v64 + 36) != 493 )
LABEL_96:
          BUG();
        v87[0] = (const char *)a3;
        v65 = (__int64 *)sub_B43CA0((__int64)a2);
        v66 = sub_B6E160(v65, 0x1EDu, (__int64)v87, 1);
        v88 = 257;
        v67 = v66;
        v78 = *(_QWORD *)(v66 + 24);
        v68 = sub_BD2C40(88, 1u);
        v16 = (__int64)v68;
        if ( v68 )
          sub_B4A410((__int64)v68, v78, v67, (__int64)v87, 1u, 0, 0, 0);
LABEL_73:
        v17 = (unsigned __int8 *)v16;
      }
      else
      {
        v17 = 0;
        v16 = 0;
      }
      break;
    case 'V':
      v26 = a4;
      if ( (a2[7] & 0x40) != 0 )
        v27 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v27 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v80 = a3;
      v28 = sub_1106750(a1, *((_QWORD *)v27 + 4), a3, a4);
      if ( (a2[7] & 0x40) != 0 )
        v29 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v29 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v30 = sub_1106750(a1, *((_QWORD *)v29 + 8), v80, v26);
      v88 = 257;
      v31 = v30;
      if ( (a2[7] & 0x40) != 0 )
        v32 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v32 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v81 = *(_QWORD *)v32;
      v33 = (unsigned __int8 *)sub_BD2C40(72, 3u);
      v16 = (__int64)v33;
      if ( v33 )
      {
        v74 = v81;
        v82 = v33;
        sub_B44260((__int64)v33, *(_QWORD *)(v28 + 8), 57, 3u, 0, 0);
        if ( *(_QWORD *)(v16 - 96) )
        {
          v34 = *(_QWORD *)(v16 - 88);
          **(_QWORD **)(v16 - 80) = v34;
          if ( v34 )
            *(_QWORD *)(v34 + 16) = *(_QWORD *)(v16 - 80);
        }
        *(_QWORD *)(v16 - 96) = v74;
        if ( v74 )
        {
          v35 = *(_QWORD *)(v74 + 16);
          *(_QWORD *)(v16 - 88) = v35;
          if ( v35 )
            *(_QWORD *)(v35 + 16) = v16 - 88;
          *(_QWORD *)(v16 - 80) = v74 + 16;
          *(_QWORD *)(v74 + 16) = v16 - 96;
        }
        if ( *(_QWORD *)(v16 - 64) )
        {
          v36 = *(_QWORD *)(v16 - 56);
          **(_QWORD **)(v16 - 48) = v36;
          if ( v36 )
            *(_QWORD *)(v36 + 16) = *(_QWORD *)(v16 - 48);
        }
        *(_QWORD *)(v16 - 64) = v28;
        v37 = *(_QWORD *)(v28 + 16);
        *(_QWORD *)(v16 - 56) = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = v16 - 56;
        *(_QWORD *)(v16 - 48) = v28 + 16;
        *(_QWORD *)(v28 + 16) = v16 - 64;
        if ( *(_QWORD *)(v16 - 32) )
        {
          v38 = *(_QWORD *)(v16 - 24);
          **(_QWORD **)(v16 - 16) = v38;
          if ( v38 )
            *(_QWORD *)(v38 + 16) = *(_QWORD *)(v16 - 16);
        }
        *(_QWORD *)(v16 - 32) = v31;
        if ( v31 )
        {
          v39 = *(_QWORD *)(v31 + 16);
          *(_QWORD *)(v16 - 24) = v39;
          if ( v39 )
            *(_QWORD *)(v39 + 16) = v16 - 24;
          *(_QWORD *)(v16 - 16) = v31 + 16;
          *(_QWORD *)(v31 + 16) = v16 - 32;
        }
        sub_BD6B50((unsigned __int8 *)v16, v87);
        v17 = v82;
      }
      else
      {
        v17 = 0;
      }
      break;
    case '\\':
      v55 = *(__int64 **)(a3 + 24);
      if ( (a2[7] & 0x40) != 0 )
        v56 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v56 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v57 = a4;
      v58 = *(_QWORD *)(*(_QWORD *)v56 + 8LL);
      BYTE4(v86) = *(_BYTE *)(v58 + 8) == 18;
      LODWORD(v86) = *(_DWORD *)(v58 + 32);
      v59 = sub_BCE1B0(v55, v86);
      if ( (a2[7] & 0x40) != 0 )
        v60 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v60 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v61 = sub_1106750(a1, *(_QWORD *)v60, v59, a4);
      if ( (v6[7] & 0x40) != 0 )
        v62 = (unsigned __int8 *)*((_QWORD *)v6 - 1);
      else
        v62 = &v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
      v72 = sub_1106750(a1, *((_QWORD *)v62 + 4), v59, v57);
      v85 = (void *)*((_QWORD *)v6 + 9);
      v77 = *((unsigned int *)v6 + 20);
      v88 = 257;
      v63 = sub_BD2C40(112, unk_3F1FE60);
      v16 = (__int64)v63;
      if ( v63 )
        sub_B4E9E0((__int64)v63, v61, v72, v85, v77, (__int64)v87, 0, 0);
      goto LABEL_73;
    default:
      goto LABEL_96;
  }
  v18 = (__int64)(v6 + 24);
  sub_BD6B90(v17, v6);
  v19 = (const char *)*((_QWORD *)v6 + 6);
  v87[0] = v19;
  if ( v19 )
  {
    v20 = v16 + 48;
    sub_B96E90((__int64)v87, (__int64)v19, 1);
    v21 = *(_QWORD *)(v16 + 48);
    if ( !v21 )
      goto LABEL_12;
  }
  else
  {
    v21 = *(_QWORD *)(v16 + 48);
    v20 = v16 + 48;
    if ( !v21 )
      goto LABEL_14;
  }
  sub_B91220(v20, v21);
LABEL_12:
  v22 = (unsigned __int8 *)v87[0];
  *(const char **)(v16 + 48) = v87[0];
  if ( v22 )
    sub_B976B0((__int64)v87, v22, v20);
LABEL_14:
  sub_B44220((_QWORD *)v16, v18, 0);
  v23 = *(_QWORD *)(a1 + 40);
  v87[0] = (const char *)v16;
  sub_1106230(v23 + 2096, (__int64 *)v87);
  return v16;
}
