// Function: sub_27DD6D0
// Address: 0x27dd6d0
//
void __fastcall sub_27DD6D0(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, unsigned int a6)
{
  unsigned __int64 v9; // rdi
  int v10; // eax
  unsigned __int64 v11; // rdi
  bool v12; // cf
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v16; // r13
  _QWORD *v17; // rax
  _QWORD *v18; // r9
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // eax
  int v27; // eax
  unsigned int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r14
  __int64 v41; // rcx
  __int64 v42; // rax
  unsigned int v43; // r8d
  int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rsi
  int v47; // edx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  unsigned int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // r8
  int v54; // eax
  char *v55; // rdx
  unsigned int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // r8
  int v59; // eax
  char *v60; // rdx
  __int64 v61; // rdi
  __int64 v62; // [rsp+8h] [rbp-A8h]
  __int64 v63; // [rsp+8h] [rbp-A8h]
  __int64 v64; // [rsp+8h] [rbp-A8h]
  __int64 v65; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v66; // [rsp+10h] [rbp-A0h]
  _QWORD *v67; // [rsp+10h] [rbp-A0h]
  __int64 v68; // [rsp+10h] [rbp-A0h]
  __int64 v69; // [rsp+18h] [rbp-98h]
  unsigned int v71; // [rsp+24h] [rbp-8Ch]
  int v72; // [rsp+24h] [rbp-8Ch]
  int v73; // [rsp+24h] [rbp-8Ch]
  _QWORD *v75; // [rsp+30h] [rbp-80h]
  __int64 *v76; // [rsp+30h] [rbp-80h]
  __int64 v77; // [rsp+30h] [rbp-80h]
  unsigned __int64 v79; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v80; // [rsp+48h] [rbp-68h] BYREF
  const char *v81; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v82; // [rsp+58h] [rbp-58h]
  _QWORD v83[2]; // [rsp+60h] [rbp-50h] BYREF
  char v84; // [rsp+70h] [rbp-40h]
  char v85; // [rsp+71h] [rbp-3Fh]

  v9 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == a2 + 48 )
  {
    v75 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = v9 - 24;
    v12 = (unsigned int)(v10 - 30) < 0xB;
    v13 = 0;
    if ( v12 )
      v13 = (_QWORD *)v11;
    v75 = v13;
  }
  v85 = 1;
  v81 = "select.unfold";
  v14 = *(_QWORD *)(a3 + 72);
  v84 = 3;
  v62 = v14;
  v65 = sub_AA48A0(a3);
  v15 = sub_22077B0(0x50u);
  v16 = v15;
  if ( v15 )
    sub_AA4D50(v15, v65, (__int64)&v81, v62, a3);
  sub_B43D10(v75);
  sub_B44240(v75, v16, (unsigned __int64 *)(v16 + 48), 0);
  sub_B43C20((__int64)&v81, a2);
  v63 = *(a4 - 12);
  v66 = (unsigned __int64)v81;
  v69 = (unsigned __int16)v82;
  v17 = sub_BD2C40(72, 3u);
  v18 = v17;
  if ( v17 )
  {
    v19 = v66;
    v67 = v17;
    sub_B4C9A0((__int64)v17, v16, a3, v63, 3u, (__int64)v17, v19, v69);
    v18 = v67;
  }
  v64 = (__int64)v18;
  v68 = sub_B10CD0((__int64)(a4 + 6));
  v20 = sub_B10CD0((__int64)(v75 + 6));
  sub_AE8F10(v64, v20, v68);
  LODWORD(v81) = 2;
  sub_B47C00(v64, (__int64)a4, (int *)&v81, 1);
  v21 = *(a4 - 4);
  v22 = *(_QWORD *)(a5 - 8) + 32LL * a6;
  if ( *(_QWORD *)v22 )
  {
    v23 = *(_QWORD *)(v22 + 8);
    **(_QWORD **)(v22 + 16) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
  }
  *(_QWORD *)v22 = v21;
  if ( v21 )
  {
    v24 = *(_QWORD *)(v21 + 16);
    *(_QWORD *)(v22 + 8) = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = v22 + 8;
    *(_QWORD *)(v22 + 16) = v21 + 16;
    *(_QWORD *)(v21 + 16) = v22;
  }
  v25 = *(a4 - 8);
  v26 = *(_DWORD *)(a5 + 4) & 0x7FFFFFF;
  if ( v26 == *(_DWORD *)(a5 + 72) )
  {
    v77 = *(a4 - 8);
    sub_B48D90(a5);
    v25 = v77;
    v26 = *(_DWORD *)(a5 + 4) & 0x7FFFFFF;
  }
  v27 = (v26 + 1) & 0x7FFFFFF;
  v28 = v27 | *(_DWORD *)(a5 + 4) & 0xF8000000;
  v29 = *(_QWORD *)(a5 - 8) + 32LL * (unsigned int)(v27 - 1);
  *(_DWORD *)(a5 + 4) = v28;
  if ( *(_QWORD *)v29 )
  {
    v30 = *(_QWORD *)(v29 + 8);
    **(_QWORD **)(v29 + 16) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = *(_QWORD *)(v29 + 16);
  }
  *(_QWORD *)v29 = v25;
  if ( v25 )
  {
    v31 = *(_QWORD *)(v25 + 16);
    *(_QWORD *)(v29 + 8) = v31;
    if ( v31 )
      *(_QWORD *)(v31 + 16) = v29 + 8;
    *(_QWORD *)(v29 + 16) = v25 + 16;
    *(_QWORD *)(v25 + 16) = v29;
  }
  *(_QWORD *)(*(_QWORD *)(a5 - 8) + 32LL * *(unsigned int *)(a5 + 72) + 8LL * ((*(_DWORD *)(a5 + 4) & 0x7FFFFFFu) - 1)) = v16;
  v79 = 1;
  v80 = 1;
  if ( (unsigned __int8)sub_BC8C50((__int64)a4, &v79, &v80) && v80 + v79 )
  {
    v81 = (const char *)v83;
    v82 = 0x200000000LL;
    v51 = sub_F02DD0(v79, v80 + v79);
    v52 = (unsigned int)v82;
    v53 = v51;
    v54 = v82;
    if ( (unsigned int)v82 >= (unsigned __int64)HIDWORD(v82) )
    {
      if ( HIDWORD(v82) < (unsigned __int64)(unsigned int)v82 + 1 )
      {
        v72 = v53;
        sub_C8D5F0((__int64)&v81, v83, (unsigned int)v82 + 1LL, 4u, v53, (unsigned int)v82 + 1LL);
        v52 = (unsigned int)v82;
        LODWORD(v53) = v72;
      }
      *(_DWORD *)&v81[4 * v52] = v53;
      LODWORD(v82) = v82 + 1;
    }
    else
    {
      v55 = (char *)&v81[4 * (unsigned int)v82];
      if ( v55 )
      {
        *(_DWORD *)v55 = v53;
        v54 = v82;
      }
      LODWORD(v82) = v54 + 1;
    }
    v56 = sub_F02DD0(v80, v80 + v79);
    v57 = (unsigned int)v82;
    v58 = v56;
    v59 = v82;
    if ( (unsigned int)v82 >= (unsigned __int64)HIDWORD(v82) )
    {
      if ( HIDWORD(v82) < (unsigned __int64)(unsigned int)v82 + 1 )
      {
        v73 = v58;
        sub_C8D5F0((__int64)&v81, v83, (unsigned int)v82 + 1LL, 4u, v58, (unsigned int)v82 + 1LL);
        v57 = (unsigned int)v82;
        LODWORD(v58) = v73;
      }
      *(_DWORD *)&v81[4 * v57] = v58;
      LODWORD(v82) = v82 + 1;
    }
    else
    {
      v60 = (char *)&v81[4 * (unsigned int)v82];
      if ( v60 )
      {
        *(_DWORD *)v60 = v58;
        v59 = v82;
      }
      LODWORD(v82) = v59 + 1;
    }
    v61 = sub_27DD130(a1);
    if ( v61 )
      sub_FF6650(v61, a2, (__int64)&v81);
    if ( v81 != (const char *)v83 )
      _libc_free((unsigned __int64)v81);
  }
  v32 = sub_27DD5D0(a1);
  if ( v32 )
  {
    v33 = v79;
    v34 = v80 + v79;
    if ( !(v80 + v79) )
    {
      v79 = 1;
      v34 = 2;
      v33 = 1;
      v80 = 1;
    }
    v76 = (__int64 *)v32;
    v71 = sub_F02DD0(v33, v34);
    v81 = (const char *)sub_FDD860(v76, a2);
    v35 = sub_1098D20((unsigned __int64 *)&v81, v71);
    sub_FE1040(v76, v16, v35);
  }
  sub_B43D60(a4);
  v81 = (const char *)v16;
  v83[0] = a2;
  v36 = a1[6];
  v82 = a3 & 0xFFFFFFFFFFFFFFFBLL;
  v83[1] = v16 & 0xFFFFFFFFFFFFFFFBLL;
  sub_FFDB80(v36, (unsigned __int64 *)&v81, 2, v37, v38, v39);
  v40 = *(_QWORD *)(a3 + 56);
  if ( !v40 )
LABEL_36:
    BUG();
  while ( *(_BYTE *)(v40 - 24) == 84 )
  {
    if ( a5 != v40 - 24 )
    {
      v41 = *(_QWORD *)(v40 - 32);
      v42 = 0x1FFFFFFFE0LL;
      v43 = *(_DWORD *)(v40 + 48);
      v44 = *(_DWORD *)(v40 - 20) & 0x7FFFFFF;
      if ( v44 )
      {
        v45 = 0;
        do
        {
          if ( a2 == *(_QWORD *)(v41 + 32LL * v43 + 8 * v45) )
          {
            v42 = 32 * v45;
            goto LABEL_43;
          }
          ++v45;
        }
        while ( v44 != (_DWORD)v45 );
        v46 = *(_QWORD *)(v41 + 0x1FFFFFFFE0LL);
        if ( v44 == v43 )
        {
LABEL_53:
          sub_B48D90(v40 - 24);
          v41 = *(_QWORD *)(v40 - 32);
          v44 = *(_DWORD *)(v40 - 20) & 0x7FFFFFF;
        }
      }
      else
      {
LABEL_43:
        v46 = *(_QWORD *)(v41 + v42);
        if ( v44 == v43 )
          goto LABEL_53;
      }
      v47 = (v44 + 1) & 0x7FFFFFF;
      *(_DWORD *)(v40 - 20) = v47 | *(_DWORD *)(v40 - 20) & 0xF8000000;
      v48 = v41 + 32LL * (unsigned int)(v47 - 1);
      if ( *(_QWORD *)v48 )
      {
        v49 = *(_QWORD *)(v48 + 8);
        **(_QWORD **)(v48 + 16) = v49;
        if ( v49 )
          *(_QWORD *)(v49 + 16) = *(_QWORD *)(v48 + 16);
      }
      *(_QWORD *)v48 = v46;
      if ( v46 )
      {
        v50 = *(_QWORD *)(v46 + 16);
        *(_QWORD *)(v48 + 8) = v50;
        if ( v50 )
          *(_QWORD *)(v50 + 16) = v48 + 8;
        *(_QWORD *)(v48 + 16) = v46 + 16;
        *(_QWORD *)(v46 + 16) = v48;
      }
      *(_QWORD *)(*(_QWORD *)(v40 - 32)
                + 32LL * *(unsigned int *)(v40 + 48)
                + 8LL * ((*(_DWORD *)(v40 - 20) & 0x7FFFFFFu) - 1)) = v16;
    }
    v40 = *(_QWORD *)(v40 + 8);
    if ( !v40 )
      goto LABEL_36;
  }
}
