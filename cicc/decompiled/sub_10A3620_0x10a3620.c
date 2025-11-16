// Function: sub_10A3620
// Address: 0x10a3620
//
__int64 __fastcall sub_10A3620(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  _BYTE *v9; // r13
  _BYTE *v10; // r14
  char v11; // al
  _BYTE *v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // r14
  unsigned int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned int v18; // eax
  _BYTE *v19; // rax
  _BYTE *v20; // r15
  __int64 v21; // rdi
  __int64 v22; // r13
  unsigned __int8 *v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  unsigned int v28; // eax
  _BYTE *v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rdx
  _BYTE *v33; // rax
  __int64 v34; // rdx
  _BYTE *v35; // rax
  _BYTE *v36; // rdx
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rdx
  unsigned __int64 v40; // rcx
  _QWORD *v41; // rdx
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 **v44; // r10
  _BYTE *v45; // r15
  unsigned int v48; // edx
  __int64 v49; // rdi
  __int64 v50; // rdx
  unsigned int *v51; // rbx
  __int64 v52; // rdx
  unsigned int v53; // esi
  __int64 v54; // rdx
  _BYTE *v55; // rax
  __int64 v56; // rdx
  _BYTE *v57; // rax
  __int64 v58; // rdx
  _BYTE *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  _BYTE *v62; // r15
  _BYTE *v63; // rax
  _BYTE *v64; // rax
  _BYTE *v65; // rax
  __int64 v66; // [rsp+10h] [rbp-E0h]
  __int64 v67; // [rsp+18h] [rbp-D8h]
  __int64 v68; // [rsp+18h] [rbp-D8h]
  __int64 **v69; // [rsp+18h] [rbp-D8h]
  char v70; // [rsp+18h] [rbp-D8h]
  _BYTE *v71; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v72; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD *v73; // [rsp+38h] [rbp-B8h] BYREF
  _QWORD *v74; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v75; // [rsp+48h] [rbp-A8h]
  __int64 *v76; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v77; // [rsp+58h] [rbp-98h]
  __int64 *v78; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v79; // [rsp+68h] [rbp-88h]
  __int16 v80; // [rsp+80h] [rbp-70h]
  __int64 *v81; // [rsp+90h] [rbp-60h] BYREF
  _QWORD **v82; // [rsp+98h] [rbp-58h] BYREF
  char v83; // [rsp+A0h] [rbp-50h]
  __int16 v84; // [rsp+B0h] [rbp-40h]

  v4 = *(_QWORD *)(a1 - 64);
  v5 = *(_QWORD *)(a1 - 32);
  v6 = *(_QWORD *)(v4 + 16);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
  {
    v7 = *(_QWORD *)(v5 + 16);
    if ( !v7 || *(_QWORD *)(v7 + 8) )
      return 0;
  }
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v81 = (__int64 *)&v71;
  v82 = 0;
  if ( *(_BYTE *)v5 == 42 && (unsigned __int8)sub_109D250(&v81, v5) )
  {
    v30 = v4;
    v4 = v5;
    v5 = v30;
  }
  v81 = (__int64 *)&v71;
  v82 = 0;
  if ( *(_BYTE *)v4 != 42 || !(unsigned __int8)sub_109D250(&v81, v4) )
    goto LABEL_7;
  if ( *(_BYTE *)v5 == 59
    && *(_QWORD *)(v5 - 64)
    && ((v49 = *(_QWORD *)(v5 - 32), *(_BYTE *)v49 == 17)
     || (v58 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v49 + 8) + 8LL) - 17, (unsigned int)v58 <= 1)
     && *(_BYTE *)v49 <= 0x15u
     && (v59 = sub_AD7630(v49, 0, v58)) != 0
     && *v59 == 17) )
  {
    v50 = (__int64)v71;
    v19 = (_BYTE *)v5;
    v71 = (_BYTE *)v5;
    v5 = v50;
  }
  else
  {
    v19 = v71;
  }
  if ( *v19 != 59 )
    goto LABEL_7;
  v20 = (_BYTE *)*((_QWORD *)v19 - 8);
  if ( !v20 )
    goto LABEL_7;
  v21 = *((_QWORD *)v19 - 4);
  v22 = v21 + 24;
  if ( *(_BYTE *)v21 != 17 )
  {
    v54 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v21 + 8) + 8LL) - 17;
    if ( (unsigned int)v54 > 1 )
      goto LABEL_7;
    if ( *(_BYTE *)v21 > 0x15u )
      goto LABEL_7;
    v55 = sub_AD7630(v21, 0, v54);
    if ( !v55 || *v55 != 17 )
      goto LABEL_7;
    v22 = (__int64)(v55 + 24);
  }
  v83 = 0;
  v81 = &v72;
  v82 = &v73;
  if ( *v20 == 58 )
  {
    if ( *((_QWORD *)v20 - 8) )
    {
      v72 = *((_QWORD *)v20 - 8);
      if ( (unsigned __int8)sub_991580((__int64)&v82, *((_QWORD *)v20 - 4)) )
      {
        sub_9865C0((__int64)&v76, v22);
        sub_987160((__int64)&v76, v22, v60, v61, (__int64)&v76);
        v79 = v77;
        v77 = 0;
        v78 = v76;
        v70 = sub_AAD8B0((__int64)v73, &v78);
        sub_969240((__int64 *)&v78);
        sub_969240((__int64 *)&v76);
        if ( v70 )
        {
          v62 = (_BYTE *)v72;
          v84 = 257;
          v63 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v72 + 8), v22);
          v64 = (_BYTE *)sub_A82350((unsigned int **)a2, v62, v63, (__int64)&v81);
          v84 = 259;
          v81 = (__int64 *)"sub";
          return sub_929DE0((unsigned int **)a2, (_BYTE *)v5, v64, (__int64)&v81, 0, 0);
        }
      }
    }
  }
  if ( *v20 != 57 || !*((_QWORD *)v20 - 8) )
    goto LABEL_7;
  v72 = *((_QWORD *)v20 - 8);
  v23 = (unsigned __int8 *)*((_QWORD *)v20 - 4);
  v24 = *v23;
  if ( (_BYTE)v24 == 17 )
  {
    v73 = v23 + 24;
    goto LABEL_35;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v23 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v24 <= 0x15u )
  {
    v65 = sub_AD7630((__int64)v23, 0, v24);
    if ( v65 )
    {
      if ( *v65 == 17 )
      {
        v73 = v65 + 24;
LABEL_35:
        if ( sub_AAD8B0(v22, v73) )
        {
          v84 = 257;
          sub_9865C0((__int64)&v76, v22);
          sub_987160((__int64)&v76, v22, v25, v26, v27);
          v28 = v77;
          v77 = 0;
          v79 = v28;
          v78 = v76;
          v29 = (_BYTE *)sub_10A01A0((__int64 *)a2, v72, (__int64)&v78, (__int64)&v81);
          sub_969240((__int64 *)&v78);
          sub_969240((__int64 *)&v76);
          v81 = (__int64 *)"sub";
          v84 = 259;
          return sub_929DE0((unsigned int **)a2, (_BYTE *)v5, v29, (__int64)&v81, 0, 0);
        }
      }
    }
  }
LABEL_7:
  v9 = *(_BYTE **)(a1 - 32);
  v10 = *(_BYTE **)(a1 - 64);
  if ( *v9 != 59 || !*((_QWORD *)v9 - 8) )
    goto LABEL_8;
  v31 = *((_QWORD *)v9 - 4);
  if ( *(_BYTE *)v31 == 17 )
  {
    v10 = *(_BYTE **)(a1 - 32);
    v9 = *(_BYTE **)(a1 - 64);
    goto LABEL_10;
  }
  v34 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v31 + 8) + 8LL) - 17;
  if ( (unsigned int)v34 <= 1 && *(_BYTE *)v31 <= 0x15u && (v35 = sub_AD7630(v31, 0, v34)) != 0 && *v35 == 17 )
  {
    v36 = v10;
    v11 = *v9;
    v10 = v9;
    v9 = v36;
  }
  else
  {
LABEL_8:
    v11 = *v10;
  }
  if ( v11 != 59 )
    return 0;
LABEL_10:
  v12 = (_BYTE *)*((_QWORD *)v10 - 8);
  if ( !v12 )
    return 0;
  v13 = *((_QWORD *)v10 - 4);
  v14 = v13 + 24;
  if ( *(_BYTE *)v13 != 17 )
  {
    v32 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
    if ( (unsigned int)v32 > 1 )
      return 0;
    if ( *(_BYTE *)v13 > 0x15u )
      return 0;
    v33 = sub_AD7630(v13, 0, v32);
    if ( !v33 || *v33 != 17 )
      return 0;
    v14 = (__int64)(v33 + 24);
  }
  v15 = *(_DWORD *)(v14 + 8);
  if ( v15 <= 0x40 )
  {
    _RDX = *(_QWORD *)v14;
    __asm { tzcnt   rcx, rdx }
    v48 = 64;
    if ( *(_QWORD *)v14 )
      v48 = _RCX;
    if ( v15 > v48 )
      v15 = v48;
  }
  else
  {
    v15 = sub_C44590(v14);
  }
  if ( v15 || *v12 != 57 || !*((_QWORD *)v12 - 8) )
    return 0;
  v72 = *((_QWORD *)v12 - 8);
  v16 = *((_QWORD *)v12 - 4);
  if ( *(_BYTE *)v16 == 17 )
  {
    v17 = v16 + 24;
    v73 = (_QWORD *)(v16 + 24);
    goto LABEL_19;
  }
  v56 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + 8LL) - 17;
  if ( (unsigned int)v56 > 1 )
    return 0;
  if ( *(_BYTE *)v16 > 0x15u )
    return 0;
  v57 = sub_AD7630(v16, 0, v56);
  if ( !v57 || *v57 != 17 )
    return 0;
  v17 = (__int64)(v57 + 24);
  v73 = v57 + 24;
LABEL_19:
  sub_9865C0((__int64)&v78, v17);
  sub_C46A40((__int64)&v78, 1);
  v18 = v79;
  v79 = 0;
  LODWORD(v82) = v18;
  v81 = v78;
  if ( *(_DWORD *)(v14 + 8) <= 0x40u )
  {
    if ( v78 != *(__int64 **)v14 )
      goto LABEL_21;
  }
  else if ( !sub_C43C50(v14, (const void **)&v81) )
  {
LABEL_21:
    sub_969240((__int64 *)&v81);
    sub_969240((__int64 *)&v78);
    return 0;
  }
  sub_969240((__int64 *)&v81);
  sub_969240((__int64 *)&v78);
  v80 = 257;
  v75 = *((_DWORD *)v73 + 2);
  v37 = v75;
  if ( v75 <= 0x40 )
  {
    v38 = *v73;
LABEL_55:
    v39 = ~v38;
    v40 = 0;
    if ( v37 )
      v40 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v37;
    v41 = (_QWORD *)(v40 & v39);
    v74 = v41;
    goto LABEL_58;
  }
  sub_C43780((__int64)&v74, (const void **)v73);
  v37 = v75;
  if ( v75 <= 0x40 )
  {
    v38 = (__int64)v74;
    goto LABEL_55;
  }
  sub_C43D10((__int64)&v74);
  v37 = v75;
  v41 = v74;
LABEL_58:
  v42 = v72;
  v77 = v37;
  v76 = v41;
  v75 = 0;
  v67 = sub_AD8D80(*(_QWORD *)(v72 + 8), (__int64)&v76);
  v43 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 16LL))(
          *(_QWORD *)(a2 + 80),
          29,
          v42,
          v67);
  v44 = &v81;
  v45 = (_BYTE *)v43;
  if ( !v43 )
  {
    v84 = 257;
    v45 = (_BYTE *)sub_B504D0(29, v42, v67, (__int64)&v81, 0, 0);
    (*(void (__fastcall **)(_QWORD, _BYTE *, __int64 **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v45,
      &v78,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v44 = &v81;
    v51 = *(unsigned int **)a2;
    v66 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v66 )
    {
      do
      {
        v52 = *((_QWORD *)v51 + 1);
        v53 = *v51;
        v69 = v44;
        v51 += 4;
        sub_B99FD0((__int64)v45, v53, v52);
        v44 = v69;
      }
      while ( (unsigned int *)v66 != v51 );
    }
  }
  v68 = (__int64)v44;
  sub_969240((__int64 *)&v76);
  sub_969240((__int64 *)&v74);
  v81 = (__int64 *)"sub";
  v84 = 259;
  return sub_929DE0((unsigned int **)a2, v9, v45, v68, 0, 0);
}
