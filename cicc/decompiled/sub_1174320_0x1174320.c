// Function: sub_1174320
// Address: 0x1174320
//
__int64 *__fastcall sub_1174320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  int v12; // eax
  unsigned __int64 v13; // rax
  unsigned __int8 *v14; // r15
  unsigned __int8 v15; // al
  unsigned __int8 *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int8 *v20; // r13
  unsigned __int8 *v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 *v28; // rbx
  _BYTE *v29; // r12
  char v30; // al
  _BYTE *v31; // rcx
  const char *v32; // rax
  int v33; // ebx
  int v34; // ebx
  _QWORD *v35; // rdx
  unsigned __int8 *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 *v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // rsi
  int v42; // eax
  int v43; // eax
  unsigned int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // r8
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdx
  const char *v53; // r11
  _QWORD *v54; // r10
  _QWORD *v55; // r9
  const char *v56; // r12
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 *v59; // rax
  __int64 v60; // rdx
  int v61; // eax
  int v62; // eax
  unsigned int v63; // ecx
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // r9
  int v69; // eax
  __int64 v70; // rdx
  _BYTE *v71; // rcx
  __int64 v72; // rdi
  __int64 v73; // rax
  unsigned __int64 *v74; // rdx
  unsigned __int64 *v75; // r15
  unsigned __int64 *i; // rbx
  unsigned __int64 v77; // rsi
  _QWORD *v78; // [rsp+8h] [rbp-C8h]
  _QWORD *v79; // [rsp+10h] [rbp-C0h]
  const char *v80; // [rsp+18h] [rbp-B8h]
  __int64 v81; // [rsp+20h] [rbp-B0h]
  __int64 v82; // [rsp+30h] [rbp-A0h]
  __int64 v83; // [rsp+38h] [rbp-98h]
  __int64 v84; // [rsp+38h] [rbp-98h]
  __int64 v85; // [rsp+38h] [rbp-98h]
  _BYTE *v86; // [rsp+40h] [rbp-90h]
  __int64 *v87; // [rsp+48h] [rbp-88h]
  __int64 v88; // [rsp+48h] [rbp-88h]
  __int64 v89[4]; // [rsp+50h] [rbp-80h] BYREF
  const char *v90; // [rsp+70h] [rbp-60h] BYREF
  _QWORD *v91; // [rsp+78h] [rbp-58h]
  const char *v92; // [rsp+80h] [rbp-50h]
  _QWORD *v93; // [rsp+88h] [rbp-48h]
  __int16 v94; // [rsp+90h] [rbp-40h]

  v7 = a1;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(v8 + 48);
  v10 = v8 + 48;
  v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 != v10 )
  {
    if ( !v11 )
      BUG();
    v12 = *(unsigned __int8 *)(v11 - 24);
    v10 = (unsigned int)(v12 - 30);
    if ( (unsigned int)v10 <= 0xA )
    {
      v13 = (unsigned int)(v12 - 39);
      if ( (unsigned int)v13 <= 0x38 )
      {
        v10 = 0x100060000000001LL;
        if ( _bittest64(&v10, v13) )
          return 0;
      }
    }
  }
  v14 = **(unsigned __int8 ***)(a2 - 8);
  v15 = *v14;
  if ( *v14 == 63 )
    return sub_1172510(a1, a2, v10, a4, a5, a6);
  if ( v15 == 94 )
    return sub_1172080(a1, a2);
  if ( v15 == 93 )
    return sub_1173460(a1, a2);
  if ( (unsigned int)v15 - 67 <= 0xC )
  {
    if ( (v14[7] & 0x40) != 0 )
      v16 = (unsigned __int8 *)*((_QWORD *)v14 - 1);
    else
      v16 = &v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
    v17 = *(_QWORD *)(a2 + 8);
    v86 = 0;
    v18 = *(_QWORD *)(*(_QWORD *)v16 + 8LL);
    v19 = v18;
    if ( *(_BYTE *)(v17 + 8) == 12 && *(_BYTE *)(v18 + 8) == 12 && !(unsigned __int8)sub_F0C890(a1, v17, v18) )
      return 0;
    goto LABEL_22;
  }
  if ( (unsigned int)v15 - 42 > 0x11 && (unsigned __int8)(v15 - 82) > 1u )
    return 0;
  if ( (v14[7] & 0x40) != 0 )
    v22 = (unsigned __int8 *)*((_QWORD *)v14 - 1);
  else
    v22 = &v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
  v19 = 0;
  v86 = (_BYTE *)*((_QWORD *)v22 + 4);
  if ( *v86 <= 0x15u )
  {
LABEL_22:
    v82 = a2;
    v23 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v24 = *(_QWORD *)(a2 - 8);
      v25 = v24 + v23;
    }
    else
    {
      v24 = a2 - v23;
      v25 = a2;
    }
    v26 = sub_116D080(v24, v25, 1);
    v87 = v27;
    if ( v27 == (__int64 *)v26 )
    {
LABEL_35:
      v32 = sub_BD5D20(a2);
      v33 = *(_DWORD *)(a2 + 4);
      v90 = v32;
      v94 = 773;
      v34 = v33 & 0x7FFFFFF;
      v91 = v35;
      v92 = (const char *)&off_3F92B2E;
      if ( (v14[7] & 0x40) != 0 )
        v36 = (unsigned __int8 *)*((_QWORD *)v14 - 1);
      else
        v36 = &v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
      v88 = *(_QWORD *)(*(_QWORD *)v36 + 8LL);
      v37 = sub_BD2DA0(80);
      v38 = v37;
      if ( v37 )
      {
        sub_B44260(v37, v88, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v38 + 72) = v34;
        sub_BD6B50((unsigned __int8 *)v38, &v90);
        sub_BD2A10(v38, *(_DWORD *)(v38 + 72), 1);
      }
      if ( (v14[7] & 0x40) != 0 )
        v39 = (__int64 *)*((_QWORD *)v14 - 1);
      else
        v39 = (__int64 *)&v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
      v40 = *v39;
      v41 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72));
      v42 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
      if ( v42 == *(_DWORD *)(v38 + 72) )
      {
        v85 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72));
        sub_B48D90(v38);
        v41 = v85;
        v42 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
      }
      v43 = (v42 + 1) & 0x7FFFFFF;
      v44 = v43 | *(_DWORD *)(v38 + 4) & 0xF8000000;
      v45 = *(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v43 - 1);
      *(_DWORD *)(v38 + 4) = v44;
      if ( *(_QWORD *)v45 )
      {
        v46 = *(_QWORD *)(v45 + 8);
        **(_QWORD **)(v45 + 16) = v46;
        if ( v46 )
          *(_QWORD *)(v46 + 16) = *(_QWORD *)(v45 + 16);
      }
      *(_QWORD *)v45 = v40;
      if ( v40 )
      {
        v47 = *(_QWORD *)(v40 + 16);
        *(_QWORD *)(v45 + 8) = v47;
        if ( v47 )
          *(_QWORD *)(v47 + 16) = v45 + 8;
        *(_QWORD *)(v45 + 16) = v40 + 16;
        *(_QWORD *)(v40 + 16) = v45;
      }
      *(_QWORD *)(*(_QWORD *)(v38 - 8)
                + 32LL * *(unsigned int *)(v38 + 72)
                + 8LL * ((*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1)) = v41;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      {
        v48 = *(_QWORD *)(a2 - 8);
        v49 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
        v50 = v48;
        v51 = v48 + 32 * v49;
      }
      else
      {
        v51 = a2;
        v49 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
        v48 = a2 - 32 * v49;
        v50 = *(_QWORD *)(a2 - 8);
      }
      v52 = *(unsigned int *)(a2 + 72);
      v89[1] = v51;
      v89[0] = v48;
      v52 *= 32;
      v89[2] = v50 + v52;
      v89[3] = v52 + 8 * v49 + v50;
      sub_116D910(&v90, v89, 1);
      v53 = v90;
      v54 = v91;
      v55 = v93;
      if ( v90 != v92 && v93 != v91 )
      {
        v84 = v7;
        v56 = v92;
        do
        {
          v57 = *(_QWORD *)v53;
          v58 = *v54;
          if ( (*(_BYTE *)(*(_QWORD *)v53 + 7LL) & 0x40) != 0 )
            v59 = *(__int64 **)(v57 - 8);
          else
            v59 = (__int64 *)(v57 - 32LL * (*(_DWORD *)(v57 + 4) & 0x7FFFFFF));
          v60 = *v59;
          if ( v40 != *v59 )
            v40 = 0;
          v61 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
          if ( v61 == *(_DWORD *)(v38 + 72) )
          {
            v78 = v54;
            v79 = v55;
            v80 = v53;
            v81 = v60;
            sub_B48D90(v38);
            v54 = v78;
            v55 = v79;
            v53 = v80;
            v60 = v81;
            v61 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
          }
          v62 = (v61 + 1) & 0x7FFFFFF;
          v63 = v62 | *(_DWORD *)(v38 + 4) & 0xF8000000;
          v64 = *(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v62 - 1);
          *(_DWORD *)(v38 + 4) = v63;
          if ( *(_QWORD *)v64 )
          {
            v65 = *(_QWORD *)(v64 + 8);
            **(_QWORD **)(v64 + 16) = v65;
            if ( v65 )
              *(_QWORD *)(v65 + 16) = *(_QWORD *)(v64 + 16);
          }
          *(_QWORD *)v64 = v60;
          if ( v60 )
          {
            v66 = *(_QWORD *)(v60 + 16);
            *(_QWORD *)(v64 + 8) = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = v64 + 8;
            *(_QWORD *)(v64 + 16) = v60 + 16;
            *(_QWORD *)(v60 + 16) = v64;
          }
          ++v54;
          v53 += 32;
          *(_QWORD *)(*(_QWORD *)(v38 - 8)
                    + 32LL * *(unsigned int *)(v38 + 72)
                    + 8LL * ((*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1)) = v58;
        }
        while ( v54 != v55 && v56 != v53 );
        v7 = v84;
      }
      if ( v40 )
      {
        sub_B43C40(v38);
        sub_BD2DD0(v38);
      }
      else
      {
        v40 = v38;
        sub_B44220((_QWORD *)v38, a2 + 24, 0);
        v67 = *(_QWORD *)(v7 + 40);
        v90 = (const char *)v38;
        sub_11715E0(v67 + 2096, (__int64 *)&v90);
      }
      v69 = *v14;
      if ( (unsigned int)(v69 - 67) > 0xC )
      {
        v94 = 257;
        if ( (unsigned int)(v69 - 42) > 0x11 )
        {
          v20 = (unsigned __int8 *)sub_B52500(
                                     (unsigned int)*v14 - 29,
                                     *((_WORD *)v14 + 1) & 0x3F,
                                     v40,
                                     (__int64)v86,
                                     (__int64)&v90,
                                     v68,
                                     0,
                                     0);
          sub_116D800(v7, (__int64)v20, a2);
        }
        else
        {
          v20 = (unsigned __int8 *)sub_B504D0((unsigned int)*v14 - 29, v40, (__int64)v86, (__int64)&v90, 0, 0);
          sub_B45260(v20, **(_QWORD **)(a2 - 8), 1);
          if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          {
            v72 = *(_QWORD *)(a2 - 8);
            v82 = v72 + 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
          }
          else
          {
            v72 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
          }
          v73 = sub_116D080(v72, v82, 1);
          v75 = v74;
          for ( i = (unsigned __int64 *)v73; v75 != i; i += 4 )
          {
            v77 = *i;
            sub_B45560(v20, v77);
          }
          sub_116D800(v7, (__int64)v20, a2);
        }
      }
      else
      {
        v70 = *(_QWORD *)(a2 + 8);
        v94 = 257;
        v20 = (unsigned __int8 *)sub_B51D30((unsigned int)*v14 - 29, v40, v70, (__int64)&v90, 0, 0);
        sub_116D800(v7, (__int64)v20, a2);
      }
      return (__int64 *)v20;
    }
    v83 = v7;
    v28 = (__int64 *)v26;
    while ( 1 )
    {
      v29 = (_BYTE *)*v28;
      if ( *(_BYTE *)*v28 <= 0x1Cu
        || !(unsigned __int8)sub_BD36B0(*v28)
        || !(unsigned __int8)sub_B46250((__int64)v29, (__int64)v14, 0) )
      {
        return 0;
      }
      v30 = v29[7] & 0x40;
      if ( v19 )
      {
        if ( v30 )
          v31 = (_BYTE *)*((_QWORD *)v29 - 1);
        else
          v31 = &v29[-32 * (*((_DWORD *)v29 + 1) & 0x7FFFFFF)];
        if ( v19 != *(_QWORD *)(*(_QWORD *)v31 + 8LL) )
          return 0;
      }
      else
      {
        if ( v30 )
          v71 = (_BYTE *)*((_QWORD *)v29 - 1);
        else
          v71 = &v29[-32 * (*((_DWORD *)v29 + 1) & 0x7FFFFFF)];
        if ( v86 != *((_BYTE **)v71 + 4) )
          return 0;
      }
      v28 += 4;
      if ( v87 == v28 )
      {
        v7 = v83;
        goto LABEL_35;
      }
    }
  }
  return (__int64 *)sub_1173890(a1, a2);
}
