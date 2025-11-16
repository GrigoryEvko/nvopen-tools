// Function: sub_140B940
// Address: 0x140b940
//
__int64 __fastcall sub_140B940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r15
  unsigned __int64 v7; // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rcx
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned int v16; // ecx
  unsigned __int64 v17; // rax
  __int64 v18; // rsi
  __int16 v19; // cx
  unsigned __int64 v20; // r13
  unsigned int v21; // eax
  unsigned int v22; // eax
  unsigned __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rax
  int v26; // eax
  unsigned int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rdi
  unsigned __int64 v31; // r15
  _QWORD *v32; // rax
  __int16 v33; // cx
  unsigned __int64 v34; // r13
  unsigned int v35; // eax
  unsigned int v36; // eax
  __int64 v37; // rax
  int v38; // eax
  unsigned __int64 v39; // rax
  _QWORD *v40; // rax
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-A0h]
  __int64 v44; // [rsp+0h] [rbp-A0h]
  __int64 v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+8h] [rbp-98h]
  __int64 v47; // [rsp+8h] [rbp-98h]
  __int64 v48; // [rsp+8h] [rbp-98h]
  __int64 v49; // [rsp+8h] [rbp-98h]
  __int64 v50; // [rsp+10h] [rbp-90h]
  __int64 v51; // [rsp+10h] [rbp-90h]
  __int64 v52; // [rsp+18h] [rbp-88h]
  __int64 v53; // [rsp+18h] [rbp-88h]
  __int64 v54; // [rsp+18h] [rbp-88h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+18h] [rbp-88h]
  char v57; // [rsp+2Fh] [rbp-71h] BYREF
  unsigned __int64 v58; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-68h]
  unsigned __int64 v60; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v61; // [rsp+48h] [rbp-58h]
  unsigned __int64 v62; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v63; // [rsp+58h] [rbp-48h]
  unsigned __int64 v64; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v65; // [rsp+68h] [rbp-38h]

  v6 = *(_QWORD *)(a3 + 56);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned __int8)v7 > 0xFu || (v9 = 35454, !_bittest64(&v9, v7)) )
  {
LABEL_2:
    if ( (unsigned int)(v7 - 13) > 1 && (_DWORD)v7 != 16 || !(unsigned __int8)sub_16435F0(v6, 0) )
    {
      *(_DWORD *)(a1 + 8) = 1;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 24) = 1;
      *(_QWORD *)(a1 + 16) = 0;
      return a1;
    }
    v6 = *(_QWORD *)(a3 + 56);
  }
  v52 = *(_QWORD *)a2;
  v10 = sub_15A9FE0(*(_QWORD *)a2, v6);
  v11 = v52;
  v12 = 1;
  v13 = v10;
  while ( 2 )
  {
    LODWORD(v7) = *(unsigned __int8 *)(v6 + 8);
    switch ( (char)v7 )
    {
      case 0:
      case 8:
      case 10:
      case 12:
        v25 = *(_QWORD *)(v6 + 32);
        v6 = *(_QWORD *)(v6 + 24);
        v12 *= v25;
        continue;
      case 1:
        v14 = 16;
        break;
      case 2:
        v14 = 32;
        break;
      case 3:
      case 9:
        v14 = 64;
        break;
      case 4:
        v14 = 80;
        break;
      case 5:
      case 6:
        v14 = 128;
        break;
      case 7:
        v53 = v12;
        v24 = sub_15A9520(v11, 0);
        v12 = v53;
        v14 = (unsigned int)(8 * v24);
        break;
      case 11:
        v14 = *(_DWORD *)(v6 + 8) >> 8;
        break;
      case 13:
        v56 = v12;
        v32 = (_QWORD *)sub_15A9930(v11, v6);
        v12 = v56;
        v14 = 8LL * *v32;
        break;
      case 14:
        v43 = v12;
        v50 = v52;
        v45 = *(_QWORD *)(v6 + 24);
        v55 = *(_QWORD *)(v6 + 32);
        v27 = sub_15A9FE0(v11, v45);
        v12 = v43;
        v28 = 1;
        v29 = v45;
        v30 = v50;
        v31 = v27;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v29 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v42 = *(_QWORD *)(v29 + 32);
              v29 = *(_QWORD *)(v29 + 24);
              v28 *= v42;
              continue;
            case 1:
              v37 = 16;
              break;
            case 2:
              v37 = 32;
              break;
            case 3:
            case 9:
              v37 = 64;
              break;
            case 4:
              v37 = 80;
              break;
            case 5:
            case 6:
              v37 = 128;
              break;
            case 7:
              v49 = v28;
              v41 = sub_15A9520(v50, 0);
              v12 = v43;
              v28 = v49;
              v37 = (unsigned int)(8 * v41);
              break;
            case 0xB:
              v37 = *(_DWORD *)(v29 + 8) >> 8;
              break;
            case 0xD:
              v48 = v28;
              v40 = (_QWORD *)sub_15A9930(v50, v29);
              v12 = v43;
              v28 = v48;
              v37 = 8LL * *v40;
              break;
            case 0xE:
              v44 = v28;
              v47 = v12;
              v51 = *(_QWORD *)(v29 + 32);
              v39 = sub_12BE0A0(v30, *(_QWORD *)(v29 + 24));
              v12 = v47;
              v28 = v44;
              v37 = 8 * v51 * v39;
              break;
            case 0xF:
              v46 = v28;
              v38 = sub_15A9520(v50, *(_DWORD *)(v29 + 8) >> 8);
              v12 = v43;
              v28 = v46;
              v37 = (unsigned int)(8 * v38);
              break;
          }
          break;
        }
        v14 = 8 * v31 * v55 * ((v31 + ((unsigned __int64)(v37 * v28 + 7) >> 3) - 1) / v31);
        break;
      case 15:
        v54 = v12;
        v26 = sub_15A9520(v11, *(_DWORD *)(v6 + 8) >> 8);
        v12 = v54;
        v14 = (unsigned int)(8 * v26);
        break;
      default:
        goto LABEL_2;
    }
    break;
  }
  v15 = v13 + ((unsigned __int64)(v14 * v12 + 7) >> 3) - 1;
  v16 = *(_DWORD *)(a2 + 20);
  v59 = v16;
  v17 = v13 * (v15 / v13);
  if ( v16 > 0x40 )
  {
    sub_16A4EF0(&v58, v17, 0);
    if ( (unsigned __int8)sub_15F8BF0(a3) )
      goto LABEL_13;
LABEL_46:
    v33 = *(_WORD *)(a3 + 18);
    v63 = v59;
    v34 = (unsigned int)(1 << v33) >> 1;
    if ( v59 > 0x40 )
      sub_16A4FD0(&v62, &v58);
    else
      v62 = v58;
    sub_140B7A0((__int64)&v64, a2, (__int64)&v62, v34);
    v35 = v65;
    v65 = 0;
    *(_DWORD *)(a1 + 8) = v35;
    *(_QWORD *)a1 = v64;
    v36 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a1 + 24) = v36;
    if ( v36 > 0x40 )
    {
      sub_16A4FD0(a1 + 16, a2 + 24);
      if ( v65 > 0x40 && v64 )
        j_j___libc_free_0_0(v64);
    }
    else
    {
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 24);
    }
    if ( v63 <= 0x40 )
      goto LABEL_33;
    v23 = v62;
    if ( !v62 )
      goto LABEL_33;
    goto LABEL_32;
  }
  v58 = v17 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v16);
  if ( !(unsigned __int8)sub_15F8BF0(a3) )
    goto LABEL_46;
LABEL_13:
  v18 = *(_QWORD *)(a3 - 24);
  if ( *(_BYTE *)(v18 + 16) != 13 )
  {
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_33;
  }
  v61 = *(_DWORD *)(v18 + 32);
  if ( v61 > 0x40 )
    sub_16A4FD0(&v60, v18 + 24);
  else
    v60 = *(_QWORD *)(v18 + 24);
  if ( !(unsigned __int8)sub_140B890(a2, &v60) )
    goto LABEL_29;
  sub_16AA580(&v64, &v58, &v60, &v57);
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  v58 = v64;
  v59 = v65;
  if ( v57 )
  {
LABEL_29:
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
  }
  else
  {
    v19 = *(_WORD *)(a3 + 18);
    v63 = v65;
    v20 = (unsigned int)(1 << v19) >> 1;
    if ( v65 > 0x40 )
      sub_16A4FD0(&v62, &v58);
    else
      v62 = v64;
    sub_140B7A0((__int64)&v64, a2, (__int64)&v62, v20);
    v21 = v65;
    v65 = 0;
    *(_DWORD *)(a1 + 8) = v21;
    *(_QWORD *)a1 = v64;
    v22 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a1 + 24) = v22;
    if ( v22 > 0x40 )
    {
      sub_16A4FD0(a1 + 16, a2 + 24);
      if ( v65 > 0x40 && v64 )
        j_j___libc_free_0_0(v64);
    }
    else
    {
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 24);
    }
    if ( v63 > 0x40 && v62 )
      j_j___libc_free_0_0(v62);
  }
  if ( v61 <= 0x40 )
    goto LABEL_33;
  v23 = v60;
  if ( !v60 )
    goto LABEL_33;
LABEL_32:
  j_j___libc_free_0_0(v23);
LABEL_33:
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  return a1;
}
