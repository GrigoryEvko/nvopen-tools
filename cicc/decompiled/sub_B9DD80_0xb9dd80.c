// Function: sub_B9DD80
// Address: 0xb9dd80
//
__int64 __fastcall sub_B9DD80(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rbx
  unsigned __int8 v4; // dl
  bool v5; // cl
  unsigned int v6; // r14d
  unsigned __int8 v7; // al
  __int64 v8; // rsi
  unsigned int v9; // edi
  unsigned int v10; // r14d
  unsigned int v11; // r13d
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // r12d
  __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // rdx
  int v20; // r11d
  __int64 v21; // r10
  __int64 v22; // r8
  int v23; // eax
  unsigned int v24; // r14d
  __int64 v25; // rsi
  __int64 v26; // r15
  __int64 v27; // rbx
  __int64 v28; // r14
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int8 v32; // al
  __int64 v33; // rdx
  __int64 v34; // rbx
  __int64 v35; // rdi
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int8 v39; // al
  __int64 v40; // rdx
  __int64 v41; // r15
  __int64 v42; // rdx
  __int64 *v43; // r15
  __int64 *v44; // r12
  _QWORD *v45; // rax
  __int64 v46; // rdx
  unsigned __int64 v47; // rdx
  __int64 *v48; // rdi
  __int64 v49; // r12
  unsigned int v51; // r13d
  __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 *v55; // rax
  __int64 *i; // rdx
  __int64 *v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // [rsp+8h] [rbp-E8h]
  __int64 v61; // [rsp+10h] [rbp-E0h]
  __int64 v62; // [rsp+18h] [rbp-D8h]
  __int64 v63; // [rsp+18h] [rbp-D8h]
  __int64 v64; // [rsp+20h] [rbp-D0h]
  __int64 v65; // [rsp+20h] [rbp-D0h]
  unsigned int v66; // [rsp+28h] [rbp-C8h]
  unsigned int v67; // [rsp+34h] [rbp-BCh]
  unsigned int v68; // [rsp+38h] [rbp-B8h]
  __int64 v69; // [rsp+38h] [rbp-B8h]
  __int64 v70; // [rsp+38h] [rbp-B8h]
  _QWORD *v71; // [rsp+38h] [rbp-B8h]
  __int64 v72; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v73; // [rsp+48h] [rbp-A8h]
  __int64 v74; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v75; // [rsp+58h] [rbp-98h]
  __int64 *v76; // [rsp+60h] [rbp-90h] BYREF
  __int64 v77; // [rsp+68h] [rbp-88h]
  _BYTE v78[32]; // [rsp+70h] [rbp-80h] BYREF
  __int64 *v79; // [rsp+90h] [rbp-60h] BYREF
  __int64 v80; // [rsp+98h] [rbp-58h]
  __int64 v81[10]; // [rsp+A0h] [rbp-50h] BYREF

  if ( !a1 )
    return 0;
  v2 = a2;
  if ( !a2 )
    return 0;
  v3 = a1;
  if ( a1 == a2 )
    return a1;
  v4 = *(_BYTE *)(a1 - 16);
  v76 = (__int64 *)v78;
  v77 = 0x400000000LL;
  v5 = (v4 & 2) != 0;
  if ( (v4 & 2) != 0 )
    v6 = *(_DWORD *)(a1 - 24);
  else
    v6 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  v7 = *(_BYTE *)(a2 - 16);
  v68 = v6 >> 1;
  v8 = (v7 & 2) != 0;
  if ( (v7 & 2) != 0 )
    v9 = *(_DWORD *)(v2 - 24);
  else
    v9 = (*(_WORD *)(v2 - 16) >> 6) & 0xF;
  v67 = v9 >> 1;
  if ( v9 >> 1 && v68 )
  {
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      if ( v5 )
        v12 = *(_QWORD *)(v3 - 32);
      else
        v12 = v3 + -16 - 8LL * ((v4 >> 2) & 0xF);
      v13 = 2 * v10;
      v60 = v12;
      v14 = 2 * v10;
      v61 = *(_QWORD *)(*(_QWORD *)(v12 + 16LL * v11) + 136LL);
      v15 = v61 + 24;
      if ( (_BYTE)v8 )
      {
        v62 = *(_QWORD *)(v2 - 32);
        v64 = *(_QWORD *)(*(_QWORD *)(v62 + 8 * v13) + 136LL);
        v16 = sub_C4C880(v15, v64 + 24);
        v17 = v64;
        v18 = v62;
        v19 = (unsigned int)(v14 + 1);
        v20 = 2 * v11;
        v21 = v61;
        v22 = v60;
        if ( v16 >= 0 )
          goto LABEL_14;
      }
      else
      {
        v63 = v2 + -16 - 8LL * ((v7 >> 2) & 0xF);
        v65 = *(_QWORD *)(*(_QWORD *)(v63 + 8 * v13) + 136LL);
        v23 = sub_C4C880(v15, v65 + 24);
        v17 = v65;
        v19 = (unsigned int)(v14 + 1);
        v18 = v63;
        if ( v23 >= 0 )
        {
LABEL_14:
          v8 = v17;
          ++v10;
          sub_B90330((__int64)&v76, v17, *(_QWORD *)(*(_QWORD *)(v18 + 8 * v19) + 136LL));
          if ( v11 >= v68 )
            goto LABEL_23;
          goto LABEL_15;
        }
        v20 = 2 * v11;
        v21 = v61;
        v22 = v60;
      }
      v8 = v21;
      ++v11;
      sub_B90330((__int64)&v76, v21, *(_QWORD *)(*(_QWORD *)(v22 + 8LL * (unsigned int)(v20 + 1)) + 136LL));
      if ( v11 >= v68 )
      {
LABEL_23:
        v66 = v10;
        goto LABEL_24;
      }
LABEL_15:
      if ( v67 <= v10 )
        goto LABEL_23;
      v7 = *(_BYTE *)(v2 - 16);
      v4 = *(_BYTE *)(v3 - 16);
      LOBYTE(v8) = (v7 & 2) != 0;
      v5 = (v4 & 2) != 0;
    }
  }
  v66 = 0;
  v11 = 0;
LABEL_24:
  if ( v68 > v11 )
  {
    v24 = v68;
    v25 = 2 * v11;
    v69 = v2;
    v26 = v3;
    v27 = 8 * v25;
    v28 = 8 * ((unsigned int)(v25 + 2) + 2LL * (v24 - 1 - v11));
    v29 = 8 * ((unsigned int)(v25 + 1) - v25);
    do
    {
      v32 = *(_BYTE *)(v26 - 16);
      if ( (v32 & 2) != 0 )
        v30 = *(_QWORD *)(v26 - 32);
      else
        v30 = v26 + -16 - 8LL * ((v32 >> 2) & 0xF);
      v33 = *(_QWORD *)(*(_QWORD *)(v30 + v29 + v27) + 136LL);
      v31 = *(_QWORD *)(v30 + v27);
      v27 += 16;
      v8 = *(_QWORD *)(v31 + 136);
      sub_B90330((__int64)&v76, v8, v33);
    }
    while ( v28 != v27 );
    v3 = v26;
    v2 = v69;
  }
  if ( v67 > v66 )
  {
    v70 = v3;
    v34 = v2;
    v35 = 2 * v66;
    v36 = 8 * v35;
    do
    {
      v39 = *(_BYTE *)(v34 - 16);
      if ( (v39 & 2) != 0 )
        v37 = *(_QWORD *)(v34 - 32);
      else
        v37 = v34 + -16 - 8LL * ((v39 >> 2) & 0xF);
      v40 = *(_QWORD *)(*(_QWORD *)(v37 + 8 * ((unsigned int)(v35 + 1) - v35) + v36) + 136LL);
      v38 = *(_QWORD *)(v37 + v36);
      v36 += 16;
      v8 = *(_QWORD *)(v38 + 136);
      sub_B90330((__int64)&v76, v8, v40);
    }
    while ( v36 != 8 * ((unsigned int)(v35 + 2) + 2LL * (v67 - 1 - v66)) );
    v3 = v70;
  }
  v41 = (unsigned int)v77;
  if ( (unsigned int)v77 > 2 )
  {
    v8 = *v76;
    if ( !(unsigned __int8)sub_B8FF20((__int64)&v76, *v76, v76[1]) )
    {
      LODWORD(v41) = v77;
      if ( (_DWORD)v77 != 2 )
        goto LABEL_40;
      goto LABEL_67;
    }
    v51 = v41 - 2;
    v52 = 16;
    v53 = 8 * v41;
    do
    {
      v76[(unsigned __int64)v52 / 8 - 2] = v76[(unsigned __int64)v52 / 8];
      v52 += 8;
    }
    while ( v53 != v52 );
    v54 = (unsigned int)v77;
    LODWORD(v41) = v77;
    if ( v51 != (unsigned __int64)(unsigned int)v77 )
    {
      if ( v51 >= (unsigned __int64)(unsigned int)v77 )
      {
        if ( v51 > (unsigned __int64)HIDWORD(v77) )
        {
          v8 = (__int64)v78;
          sub_C8D5F0(&v76, v78, v51, 8);
          v54 = (unsigned int)v77;
        }
        v55 = &v76[v54];
        for ( i = &v76[v51]; i != v55; ++v55 )
        {
          if ( v55 )
            *v55 = 0;
        }
      }
      LODWORD(v77) = v51;
      LODWORD(v41) = v51;
    }
  }
  if ( (_DWORD)v41 != 2 )
  {
LABEL_40:
    v79 = v81;
    v42 = (unsigned int)v41;
    v80 = 0x400000000LL;
    if ( (unsigned int)v41 > 4 )
    {
      v8 = (__int64)v81;
      sub_C8D5F0(&v79, v81, (unsigned int)v41, 8);
      v42 = (unsigned int)v77;
    }
    v43 = v76;
    v44 = &v76[v42];
    if ( v44 == v76 )
    {
      v47 = (unsigned int)v80;
    }
    else
    {
      do
      {
        v45 = sub_B98A20(*v43, v8);
        v46 = (unsigned int)v80;
        if ( (unsigned __int64)(unsigned int)v80 + 1 > HIDWORD(v80) )
        {
          v8 = (__int64)v81;
          v71 = v45;
          sub_C8D5F0(&v79, v81, (unsigned int)v80 + 1LL, 8);
          v46 = (unsigned int)v80;
          v45 = v71;
        }
        ++v43;
        v79[v46] = (__int64)v45;
        v47 = (unsigned int)(v80 + 1);
        LODWORD(v80) = v80 + 1;
      }
      while ( v44 != v43 );
    }
    v8 = (__int64)v79;
    v48 = (__int64 *)(*(_QWORD *)(v3 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(v3 + 8) & 4) != 0 )
      v48 = (__int64 *)*v48;
    v49 = sub_B9C770(v48, v79, (__int64 *)v47, 0, 1);
    if ( v79 != v81 )
      _libc_free(v79, v8);
    goto LABEL_50;
  }
LABEL_67:
  v57 = v76;
  v58 = v76[1];
  v75 = *(_DWORD *)(v58 + 32);
  if ( v75 > 0x40 )
  {
    sub_C43780(&v74, v58 + 24);
    v57 = v76;
  }
  else
  {
    v74 = *(_QWORD *)(v58 + 24);
  }
  v59 = *v57;
  v73 = *(_DWORD *)(*v57 + 32);
  if ( v73 > 0x40 )
    sub_C43780(&v72, v59 + 24);
  else
    v72 = *(_QWORD *)(v59 + 24);
  v8 = (__int64)&v72;
  sub_AADC30((__int64)&v79, (__int64)&v72, &v74);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  if ( !sub_AAF760((__int64)&v79) )
  {
    sub_969240(v81);
    sub_969240((__int64 *)&v79);
    LODWORD(v41) = v77;
    goto LABEL_40;
  }
  v49 = 0;
  sub_969240(v81);
  sub_969240((__int64 *)&v79);
LABEL_50:
  if ( v76 != (__int64 *)v78 )
    _libc_free(v76, v8);
  return v49;
}
