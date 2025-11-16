// Function: sub_12AA9B0
// Address: 0x12aa9b0
//
__int64 __fastcall sub_12AA9B0(__int64 a1, _QWORD *a2, int a3, unsigned __int64 *a4)
{
  __int64 v8; // rsi
  char *v9; // r12
  unsigned __int64 i; // rax
  __int64 v11; // r15
  __int64 v12; // rdx
  char *v13; // r10
  char v14; // al
  unsigned int v15; // eax
  __int64 v16; // r10
  __int64 v17; // rax
  _QWORD *v18; // r15
  __int64 v19; // rdi
  unsigned __int64 *v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  _QWORD *v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r14
  unsigned int v28; // r12d
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  char *v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // r10
  __int64 v37; // rax
  __int64 *v38; // r15
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // rax
  unsigned __int64 *v44; // r12
  __int64 v45; // rax
  unsigned __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rsi
  char *v49; // [rsp+8h] [rbp-A8h]
  __int64 v50; // [rsp+8h] [rbp-A8h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  unsigned int v53; // [rsp+18h] [rbp-98h]
  int v54; // [rsp+18h] [rbp-98h]
  __int64 v55; // [rsp+18h] [rbp-98h]
  __int64 v56; // [rsp+18h] [rbp-98h]
  int v57; // [rsp+18h] [rbp-98h]
  _DWORD *v58; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v59; // [rsp+20h] [rbp-90h]
  unsigned int v60; // [rsp+28h] [rbp-88h]
  __int64 v61; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v62[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v63; // [rsp+50h] [rbp-60h]
  _BYTE v64[16]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v65; // [rsp+70h] [rbp-40h]

  v8 = *(_QWORD *)(a4[9] + 16);
  v52 = *(_QWORD *)(v8 + 16);
  v9 = sub_128F980((__int64)a2, v8);
  for ( i = *a4; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v60 = *(_DWORD *)(*(_QWORD *)v9 + 8LL) >> 8;
  v11 = sub_1644900(a2[5], 8LL * *(_QWORD *)(i + 128));
  v12 = sub_1647190(v11, v60);
  v63 = 257;
  if ( v12 != *(_QWORD *)v9 )
  {
    if ( (unsigned __int8)v9[16] > 0x10u )
    {
      v65 = 257;
      v9 = (char *)sub_15FDBD0(47, v9, v12, v64, 0);
      v30 = a2[7];
      if ( v30 )
      {
        v59 = (unsigned __int64 *)a2[8];
        sub_157E9D0(v30 + 40, v9);
        v31 = *v59;
        v32 = *((_QWORD *)v9 + 3) & 7LL;
        *((_QWORD *)v9 + 4) = v59;
        v31 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v9 + 3) = v31 | v32;
        *(_QWORD *)(v31 + 8) = v9 + 24;
        *v59 = *v59 & 7 | (unsigned __int64)(v9 + 24);
      }
      sub_164B780(v9, v62);
      v33 = a2[6];
      if ( v33 )
      {
        v61 = a2[6];
        sub_1623A60(&v61, v33, 2);
        v34 = v9 + 48;
        if ( *((_QWORD *)v9 + 6) )
        {
          sub_161E7C0(v9 + 48);
          v34 = v9 + 48;
        }
        v35 = v61;
        *((_QWORD *)v9 + 6) = v61;
        if ( v35 )
          sub_1623210(&v61, v35, v34);
      }
    }
    else
    {
      LODWORD(v9) = sub_15A46C0(47, v9, v12, 0);
    }
  }
  v58 = (_DWORD *)a4 + 9;
  v13 = sub_128F980((__int64)a2, v52);
  v14 = *(_BYTE *)(*(_QWORD *)v13 + 8LL);
  if ( v14 == 15 )
  {
    LODWORD(v16) = sub_128B420((__int64)a2, v13, 0, v11, 0, 0, v58);
  }
  else
  {
    if ( v14 != 11 )
      sub_127B550("unexpected: a non-integer and non-pointer type was used with atomic builtin!", v58, 1);
    v49 = v13;
    v53 = ((__int64 (*)(void))sub_16431D0)();
    v15 = sub_16431D0(v11);
    LODWORD(v16) = (_DWORD)v49;
    if ( v53 > v15 )
      sub_127B550("unexpected: Integer type too small!", v58, 1);
    v63 = 257;
    if ( v11 != *(_QWORD *)v49 )
    {
      if ( (unsigned __int8)v49[16] > 0x10u )
      {
        v65 = 257;
        v36 = sub_15FDED0(v49, v11, v64, 0);
        v37 = a2[7];
        if ( v37 )
        {
          v38 = (__int64 *)a2[8];
          v55 = v36;
          sub_157E9D0(v37 + 40, v36);
          v36 = v55;
          v39 = *v38;
          v40 = *(_QWORD *)(v55 + 24);
          *(_QWORD *)(v55 + 32) = v38;
          v39 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v55 + 24) = v39 | v40 & 7;
          *(_QWORD *)(v39 + 8) = v55 + 24;
          *v38 = *v38 & 7 | (v55 + 24);
        }
        v56 = v36;
        sub_164B780(v36, v62);
        v41 = a2[6];
        LODWORD(v16) = v56;
        if ( v41 )
        {
          v50 = v56;
          v61 = a2[6];
          sub_1623A60(&v61, v41, 2);
          v16 = v56;
          if ( *(_QWORD *)(v56 + 48) )
          {
            sub_161E7C0(v56 + 48);
            v16 = v56;
          }
          v42 = v61;
          *(_QWORD *)(v16 + 48) = v61;
          if ( v42 )
          {
            v57 = v16;
            sub_1623210(&v61, v42, v50 + 48);
            LODWORD(v16) = v57;
          }
        }
      }
      else
      {
        LODWORD(v16) = sub_15A4620(v49, v11);
      }
    }
  }
  v54 = v16;
  v65 = 257;
  v17 = sub_1648A60(64, 2);
  v18 = (_QWORD *)v17;
  if ( v17 )
    sub_15F9C10(v17, a3, (_DWORD)v9, v54, 2, 1, 0);
  v19 = a2[7];
  if ( v19 )
  {
    v20 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v19 + 40, v18);
    v21 = v18[3];
    v22 = *v20;
    v18[4] = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    v18[3] = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v18 + 3;
    *v20 = *v20 & 7 | (unsigned __int64)(v18 + 3);
  }
  sub_164B780(v18, v64);
  v23 = a2[6];
  if ( v23 )
  {
    v62[0] = a2[6];
    sub_1623A60(v62, v23, 2);
    v24 = v18 + 6;
    if ( v18[6] )
    {
      sub_161E7C0(v18 + 6);
      v24 = v18 + 6;
    }
    v25 = v62[0];
    v18[6] = v62[0];
    if ( v25 )
      sub_1623210(v62, v25, v24);
  }
  v26 = sub_127A030(a2[4] + 8LL, *a4, 0);
  v27 = v26;
  if ( *(_BYTE *)(v26 + 8) == 15 )
  {
    v65 = 257;
    v18 = (_QWORD *)sub_12AA3B0(a2 + 6, 0x2Eu, (__int64)v18, v26, (__int64)v64);
  }
  else
  {
    if ( *(_BYTE *)(*v18 + 8LL) != 11 )
      sub_127B550("unexpected: a non-integer and non-pointer type was used with atomic builtin!", v58, 1);
    v28 = sub_16431D0(*v18);
    if ( v28 < (unsigned int)sub_16431D0(v27) )
      sub_127B550("unexpected: Integer type too small!", v58, 1);
    v63 = 257;
    if ( v27 != *v18 )
    {
      if ( *((_BYTE *)v18 + 16) > 0x10u )
      {
        v65 = 257;
        v18 = (_QWORD *)sub_15FDF30(v18, v27, v64, 0);
        v43 = a2[7];
        if ( v43 )
        {
          v44 = (unsigned __int64 *)a2[8];
          sub_157E9D0(v43 + 40, v18);
          v45 = v18[3];
          v46 = *v44;
          v18[4] = v44;
          v46 &= 0xFFFFFFFFFFFFFFF8LL;
          v18[3] = v46 | v45 & 7;
          *(_QWORD *)(v46 + 8) = v18 + 3;
          *v44 = *v44 & 7 | (unsigned __int64)(v18 + 3);
        }
        sub_164B780(v18, v62);
        v47 = a2[6];
        if ( v47 )
        {
          v61 = a2[6];
          sub_1623A60(&v61, v47, 2);
          if ( v18[6] )
            sub_161E7C0(v18 + 6);
          v48 = v61;
          v18[6] = v61;
          if ( v48 )
            sub_1623210(&v61, v48, v18 + 6);
        }
      }
      else
      {
        v18 = (_QWORD *)sub_15A4670(v18, v27);
      }
    }
  }
  *(_QWORD *)a1 = v18;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
