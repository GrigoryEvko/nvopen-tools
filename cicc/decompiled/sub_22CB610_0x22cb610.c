// Function: sub_22CB610
// Address: 0x22cb610
//
__int64 __fastcall sub_22CB610(
        __int64 a1,
        unsigned __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6)
{
  __int64 v6; // r10
  unsigned __int64 v7; // rax
  int v9; // ecx
  __int64 v10; // rbx
  unsigned __int64 v11; // r14
  __int64 v12; // r10
  __int64 v13; // rcx
  __int64 v14; // r9
  int v15; // eax
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v19; // rdx
  unsigned __int8 v20; // r14
  __int64 v21; // r10
  __int64 v22; // rbx
  unsigned int v23; // r12d
  __int64 v24; // r13
  const void *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rsi
  unsigned int v29; // eax
  unsigned __int64 v30; // rax
  unsigned int v31; // eax
  unsigned int v32; // eax
  unsigned int v33; // eax
  unsigned int v34; // eax
  _QWORD *v35; // rax
  __int64 v36; // rax
  unsigned __int8 *v37; // rax
  int v38; // eax
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rcx
  unsigned int v44; // r8d
  char v45; // al
  char *v46; // r9
  __int64 *v47; // r11
  char v48; // al
  unsigned int v49; // eax
  __int64 v50; // r14
  unsigned __int8 *v51; // rdx
  __int64 *v52; // rsi
  __int64 v53; // r10
  unsigned __int64 v54; // [rsp+8h] [rbp-148h]
  __int64 *v55; // [rsp+20h] [rbp-130h]
  __int64 v56; // [rsp+28h] [rbp-128h]
  __int64 v57; // [rsp+30h] [rbp-120h]
  __int64 v58; // [rsp+30h] [rbp-120h]
  char v59; // [rsp+38h] [rbp-118h]
  __int64 v60; // [rsp+38h] [rbp-118h]
  __int64 v61; // [rsp+38h] [rbp-118h]
  unsigned __int8 v62; // [rsp+38h] [rbp-118h]
  __int64 v63; // [rsp+38h] [rbp-118h]
  unsigned __int8 v64; // [rsp+40h] [rbp-110h]
  __int64 *v65; // [rsp+40h] [rbp-110h]
  __int64 v66; // [rsp+40h] [rbp-110h]
  __int64 v67; // [rsp+48h] [rbp-108h]
  __int64 v68; // [rsp+48h] [rbp-108h]
  __int64 v69; // [rsp+48h] [rbp-108h]
  const void *v71; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v72; // [rsp+68h] [rbp-E8h]
  const void *v73; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v74; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v75; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int v76; // [rsp+88h] [rbp-C8h]
  unsigned __int64 v77; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v78; // [rsp+98h] [rbp-B8h] BYREF
  unsigned __int64 v79; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v80; // [rsp+A8h] [rbp-A8h]
  const void *v81; // [rsp+C0h] [rbp-90h] BYREF
  unsigned __int64 v82; // [rsp+C8h] [rbp-88h] BYREF
  unsigned __int64 v83; // [rsp+D0h] [rbp-80h] BYREF
  unsigned __int64 v84; // [rsp+D8h] [rbp-78h] BYREF
  unsigned int v85; // [rsp+E0h] [rbp-70h]
  char v86; // [rsp+E8h] [rbp-68h]
  const void *v87; // [rsp+F0h] [rbp-60h] BYREF
  unsigned int v88; // [rsp+F8h] [rbp-58h]
  unsigned __int64 v89; // [rsp+100h] [rbp-50h] BYREF
  unsigned int v90; // [rsp+108h] [rbp-48h]
  char v91; // [rsp+118h] [rbp-38h]

  v6 = a4 + 48;
  v7 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a4 + 48 == v7 )
    goto LABEL_17;
  if ( !v7 )
    goto LABEL_111;
  v9 = *(unsigned __int8 *)(v7 - 24);
  if ( (unsigned int)(v9 - 30) > 0xA )
LABEL_17:
    BUG();
  v10 = a5;
  v11 = v7;
  if ( (_BYTE)v9 == 31 && (*(_DWORD *)(v7 - 20) & 0x7FFFFFF) == 3 )
  {
    v19 = *(_QWORD *)(v7 - 56);
    if ( v19 != *(_QWORD *)(v7 - 88) )
    {
      v20 = a5 == v19;
      if ( a3 == *(__int64 **)(v7 - 120) )
      {
        v35 = (_QWORD *)sub_BD5C60((__int64)a3);
        v36 = sub_BCB2A0(v35);
        v37 = (unsigned __int8 *)sub_ACD640(v36, v20, 0);
        LOWORD(v87) = 0;
        sub_22C0310((__int64)&v87, v37, 0);
        sub_22C0650(a1, (unsigned __int8 *)&v87);
        *(_BYTE *)(a1 + 40) = 1;
        sub_22C0090((unsigned __int8 *)&v87);
        return a1;
      }
      v60 = v6;
      v67 = *(_QWORD *)(v7 - 120);
      sub_22C9ED0((__int64)&v81, a2, (__int64)a3, v67, v20, a6, 0);
      v21 = v60;
      if ( !v86 )
      {
        *(_BYTE *)(a1 + 40) = 0;
        return a1;
      }
      if ( (_BYTE)v81 != 6 )
      {
        v22 = a1;
        *(_BYTE *)(a1 + 40) = 0;
LABEL_23:
        sub_22C0650(v22, (unsigned __int8 *)&v81);
        *(_BYTE *)(v22 + 40) = 1;
        if ( v86 )
        {
          v86 = 0;
          sub_22C0090((unsigned __int8 *)&v81);
        }
        return a1;
      }
      v38 = *(unsigned __int8 *)a3;
      if ( (unsigned __int8)v38 <= 0x1Cu )
        goto LABEL_79;
      if ( *(_BYTE *)(a3[1] + 8) != 12 )
        goto LABEL_79;
      v40 = (unsigned int)(v38 - 42);
      v61 = v67;
      v64 = v20;
      if ( (unsigned __int8)v40 > 0x36u )
        goto LABEL_79;
      v41 = 0x40003FFE03FFFFLL;
      if ( !_bittest64(&v41, v40) )
        goto LABEL_79;
      v69 = v21;
      v54 = sub_AA4E30(v10);
      v45 = sub_22BDCC0((char *)a3, v61, v42, v43, v44);
      v21 = v69;
      LODWORD(a5) = v20;
      if ( v45 )
      {
        v77 = v20;
        LODWORD(v78) = 1;
        sub_22C0930(&v87, v46, v47, (__int64)&v77, v54);
        sub_22BE8C0((unsigned __int8 *)&v81, (unsigned __int8 *)&v87);
        sub_22C0090((unsigned __int8 *)&v87);
        sub_969240((__int64 *)&v77);
        v21 = v69;
        goto LABEL_88;
      }
      v49 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
      if ( v49 )
      {
        v50 = 0;
        v62 = v64;
        v58 = (__int64)v47;
        while ( 1 )
        {
          if ( (*((_BYTE *)a3 + 7) & 0x40) != 0 )
            v51 = (unsigned __int8 *)*(a3 - 1);
          else
            v51 = (unsigned __int8 *)&a3[-4 * v49];
          v65 = *(__int64 **)&v51[32 * v50];
          sub_22C9ED0((__int64)&v87, a2, (__int64)v65, v58, v62, 0, 0);
          sub_22C0650((__int64)&v77, (unsigned __int8 *)&v87);
          if ( v91 )
          {
            v91 = 0;
            sub_22C0090((unsigned __int8 *)&v87);
          }
          if ( (_BYTE)v77 == 2 )
          {
            if ( *v78 == 17 )
            {
              v52 = (__int64 *)(v78 + 24);
              v63 = v69;
LABEL_106:
              sub_9865C0((__int64)&v73, (__int64)v52);
              LOBYTE(v75) = 1;
              sub_22C0930(&v87, (char *)a3, v65, (__int64)&v73, v54);
              sub_22BE8C0((unsigned __int8 *)&v81, (unsigned __int8 *)&v87);
              sub_22C0090((unsigned __int8 *)&v87);
              v53 = v63;
              if ( (_BYTE)v75 )
              {
                LOBYTE(v75) = 0;
                sub_969240((__int64 *)&v73);
                v53 = v63;
              }
              v66 = v53;
              sub_22C0090((unsigned __int8 *)&v77);
              v21 = v66;
              break;
            }
          }
          else if ( (unsigned __int8)(v77 - 4) <= 1u && sub_9876C0((__int64 *)&v78) )
          {
            v63 = v69;
            v52 = sub_9876C0((__int64 *)&v78);
            goto LABEL_106;
          }
          ++v50;
          sub_22C0090((unsigned __int8 *)&v77);
          v49 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
          if ( v49 <= (unsigned int)v50 )
          {
            v21 = v69;
            break;
          }
        }
      }
LABEL_88:
      v48 = v86;
      if ( (_BYTE)v81 != 6 )
      {
        v22 = a1;
        *(_BYTE *)(a1 + 40) = 0;
        if ( !v48 )
          return a1;
        goto LABEL_23;
      }
      if ( v86 )
      {
LABEL_79:
        v68 = v21;
        v86 = 0;
        sub_22C0090((unsigned __int8 *)&v81);
        v21 = v68;
      }
      v39 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v11 = v39;
      if ( v21 == v39 || !v39 )
        goto LABEL_111;
    }
  }
  if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_111:
    BUG();
  if ( *(_BYTE *)(v11 - 24) != 32 )
    goto LABEL_15;
  v12 = *(_QWORD *)(v11 - 32);
  v13 = (__int64)a3;
  v14 = a3[1];
  v55 = *(__int64 **)v12;
  if ( !*(_QWORD *)v12 )
  {
    if ( *(_BYTE *)(v14 + 8) == 12 )
      goto LABEL_11;
LABEL_15:
    *(_WORD *)a1 = 6;
    LOWORD(v87) = 0;
    *(_BYTE *)(a1 + 40) = 1;
    sub_22C0090((unsigned __int8 *)&v87);
    return a1;
  }
  if ( *(_BYTE *)(v14 + 8) != 12 )
    goto LABEL_15;
  v13 = *(_QWORD *)v12;
  if ( a3 != v55 )
  {
LABEL_11:
    v15 = *(unsigned __int8 *)a3;
    if ( (unsigned __int8)v15 <= 0x1Cu )
      goto LABEL_15;
    v16 = (unsigned int)(v15 - 42);
    if ( (unsigned __int8)v16 > 0x36u )
      goto LABEL_15;
    v17 = 0x40003FFE03FFFFLL;
    if ( !_bittest64(&v17, v16) )
      goto LABEL_15;
    v59 = sub_22BDCC0((char *)a3, (__int64)v55, 0x40003FFE03FFFFLL, v13, a5);
    if ( !v59 )
      goto LABEL_15;
    goto LABEL_26;
  }
  v59 = 0;
LABEL_26:
  v56 = *(_QWORD *)(v12 + 32);
  sub_AADB10((__int64)&v73, *(_DWORD *)(v14 + 8) >> 8, v10 == v56);
  v57 = ((*(_DWORD *)(v11 - 20) & 0x7FFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(v11 - 20) & 0x7FFFFFFu) >> 1 == 1 )
  {
LABEL_71:
    v33 = v74;
    v74 = 0;
    LODWORD(v82) = v33;
    v81 = v73;
    v34 = v76;
    v76 = 0;
    LODWORD(v84) = v34;
    v83 = v75;
    sub_22C06B0((__int64)&v87, (__int64)&v81, 0);
    sub_22C0650(a1, (unsigned __int8 *)&v87);
    *(_BYTE *)(a1 + 40) = 1;
    sub_22C0090((unsigned __int8 *)&v87);
    sub_969240((__int64 *)&v83);
    sub_969240((__int64 *)&v81);
    goto LABEL_72;
  }
  v23 = 3;
  v24 = 0;
  while ( 1 )
  {
    v28 = *(_QWORD *)(*(_QWORD *)(v11 - 32) + 32LL * (v23 - 1));
    v29 = *(_DWORD *)(v28 + 32);
    v72 = v29;
    if ( v29 <= 0x40 )
    {
      v25 = *(const void **)(v28 + 24);
      v88 = v29;
      v71 = v25;
    }
    else
    {
      sub_C43780((__int64)&v71, (const void **)(v28 + 24));
      v88 = v72;
      if ( v72 > 0x40 )
      {
        sub_C43780((__int64)&v87, &v71);
        goto LABEL_30;
      }
    }
    v87 = v71;
LABEL_30:
    sub_AADBC0((__int64)&v77, (__int64 *)&v87);
    if ( v88 > 0x40 && v87 )
      j_j___libc_free_0_0((unsigned __int64)v87);
    if ( v59 )
      break;
LABEL_34:
    v26 = 32LL * v23;
    if ( (_DWORD)v24 == -2 )
      v26 = 32;
    v27 = *(_QWORD *)(v11 - 32);
    if ( v10 == v56 )
    {
      if ( *(_QWORD *)(v27 + v26) != v10 && a3 == v55 )
      {
        sub_ABB6C0((__int64)&v87, (__int64)&v73, (__int64)&v77);
        if ( v74 <= 0x40 )
          goto LABEL_62;
        goto LABEL_67;
      }
    }
    else if ( v10 == *(_QWORD *)(v27 + v26) )
    {
      sub_AB3510((__int64)&v87, (__int64)&v73, (__int64)&v77, 0);
      if ( v74 <= 0x40 )
      {
LABEL_62:
        v73 = v87;
        v31 = v88;
        v88 = 0;
        v74 = v31;
        if ( v76 > 0x40 && v75 )
          j_j___libc_free_0_0(v75);
        v75 = v89;
        v32 = v90;
        v90 = 0;
        v76 = v32;
        sub_969240((__int64 *)&v89);
        sub_969240((__int64 *)&v87);
        goto LABEL_38;
      }
LABEL_67:
      if ( v73 )
        j_j___libc_free_0_0((unsigned __int64)v73);
      goto LABEL_62;
    }
LABEL_38:
    if ( v80 > 0x40 && v79 )
      j_j___libc_free_0_0(v79);
    if ( (unsigned int)v78 > 0x40 && v77 )
      j_j___libc_free_0_0(v77);
    if ( v72 > 0x40 && v71 )
      j_j___libc_free_0_0((unsigned __int64)v71);
    ++v24;
    v23 += 2;
    if ( v57 == v24 )
      goto LABEL_71;
  }
  v30 = sub_AA4E30(v10);
  sub_22C0930(&v81, (char *)a3, v55, (__int64)&v71, v30);
  if ( (_BYTE)v81 != 6 )
  {
    if ( (unsigned int)v78 <= 0x40 && (unsigned int)v83 <= 0x40 )
    {
      LODWORD(v78) = v83;
      v77 = v82;
    }
    else
    {
      sub_C43990((__int64)&v77, (__int64)&v82);
    }
    if ( v80 <= 0x40 && v85 <= 0x40 )
    {
      v80 = v85;
      v79 = v84;
    }
    else
    {
      sub_C43990((__int64)&v79, (__int64)&v84);
    }
    sub_22C0090((unsigned __int8 *)&v81);
    goto LABEL_34;
  }
  LOWORD(v87) = 0;
  *(_WORD *)a1 = 6;
  *(_BYTE *)(a1 + 40) = 1;
  sub_22C0090((unsigned __int8 *)&v87);
  sub_22C0090((unsigned __int8 *)&v81);
  sub_969240((__int64 *)&v79);
  sub_969240((__int64 *)&v77);
  sub_969240((__int64 *)&v71);
LABEL_72:
  sub_969240((__int64 *)&v75);
  sub_969240((__int64 *)&v73);
  return a1;
}
