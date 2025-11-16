// Function: sub_182E680
// Address: 0x182e680
//
_QWORD *__fastcall sub_182E680(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // r15
  __int64 **v15; // rax
  __int64 v16; // rsi
  __int64 **v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // r10
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 **v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rax
  _QWORD *v28; // r14
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rsi
  __int64 v45; // rdx
  unsigned __int8 *v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rsi
  __int64 v53; // rdx
  unsigned __int8 *v54; // rsi
  __int64 v55; // rax
  __int64 v56; // rdi
  unsigned __int64 *v57; // rbx
  __int64 v58; // rax
  unsigned __int64 v59; // rcx
  __int64 v60; // rsi
  __int64 v61; // rsi
  unsigned __int8 *v62; // rsi
  __int64 *v63; // [rsp+8h] [rbp-B8h]
  __int64 v64; // [rsp+10h] [rbp-B0h]
  int v65; // [rsp+18h] [rbp-A8h]
  __int64 **v66; // [rsp+18h] [rbp-A8h]
  __int64 *v67; // [rsp+18h] [rbp-A8h]
  __int64 *v68; // [rsp+18h] [rbp-A8h]
  __int64 v69; // [rsp+28h] [rbp-98h] BYREF
  __int64 v70[2]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v71; // [rsp+40h] [rbp-80h]
  __int64 v72[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v73; // [rsp+60h] [rbp-60h]
  _QWORD v74[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v75; // [rsp+80h] [rbp-40h]

  v10 = a3;
  v11 = sub_182C650(a3);
  v12 = (_QWORD *)a2[3];
  v13 = (unsigned int)(1 << *(_DWORD *)(a1 + 224));
  v14 = (unsigned int)-(int)v13 & (v13 + v11 - 1);
  v73 = 257;
  v15 = (__int64 **)sub_1643330(v12);
  if ( v15 != *(__int64 ***)a4 )
  {
    if ( *(_BYTE *)(a4 + 16) > 0x10u )
    {
      LOWORD(v75) = 257;
      v31 = sub_15FDBD0(36, a4, (__int64)v15, (__int64)v74, 0);
      v32 = a2[1];
      a4 = v31;
      if ( v32 )
      {
        v67 = (__int64 *)a2[2];
        sub_157E9D0(v32 + 40, v31);
        v33 = *v67;
        v34 = *(_QWORD *)(a4 + 24) & 7LL;
        *(_QWORD *)(a4 + 32) = v67;
        v33 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a4 + 24) = v33 | v34;
        *(_QWORD *)(v33 + 8) = a4 + 24;
        *v67 = *v67 & 7 | (a4 + 24);
      }
      sub_164B780(a4, v72);
      v35 = *a2;
      if ( *a2 )
      {
        v70[0] = *a2;
        sub_1623A60((__int64)v70, v35, 2);
        v36 = *(_QWORD *)(a4 + 48);
        v37 = a4 + 48;
        if ( v36 )
        {
          sub_161E7C0(a4 + 48, v36);
          v37 = a4 + 48;
        }
        v38 = (unsigned __int8 *)v70[0];
        *(_QWORD *)(a4 + 48) = v70[0];
        if ( v38 )
          sub_1623210((__int64)v70, v38, v37);
      }
    }
    else
    {
      a4 = sub_15A46C0(36, (__int64 ***)a4, v15, 0);
    }
  }
  if ( byte_4FAA340 )
  {
    v16 = *(_QWORD *)(a1 + 248);
    v17 = *(__int64 ***)v10;
    v73 = 257;
    v71 = 257;
    if ( (__int64 **)v16 != v17 )
    {
      if ( *(_BYTE *)(v10 + 16) > 0x10u )
      {
        LOWORD(v75) = 257;
        v39 = sub_15FDFF0(v10, v16, (__int64)v74, 0);
        v40 = a2[1];
        v10 = v39;
        if ( v40 )
        {
          v68 = (__int64 *)a2[2];
          sub_157E9D0(v40 + 40, v39);
          v41 = *v68;
          v42 = *(_QWORD *)(v10 + 24) & 7LL;
          *(_QWORD *)(v10 + 32) = v68;
          v41 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v10 + 24) = v41 | v42;
          *(_QWORD *)(v41 + 8) = v10 + 24;
          *v68 = *v68 & 7 | (v10 + 24);
        }
        sub_164B780(v10, v70);
        v43 = *a2;
        v19 = v74;
        if ( *a2 )
        {
          v69 = *a2;
          sub_1623A60((__int64)&v69, v43, 2);
          v44 = *(_QWORD *)(v10 + 48);
          v45 = v10 + 48;
          v19 = v74;
          if ( v44 )
          {
            sub_161E7C0(v10 + 48, v44);
            v19 = v74;
            v45 = v10 + 48;
          }
          v46 = (unsigned __int8 *)v69;
          *(_QWORD *)(v10 + 48) = v69;
          if ( v46 )
          {
            sub_1623210((__int64)&v69, v46, v45);
            LODWORD(v19) = (unsigned int)v74;
          }
        }
        v17 = *(__int64 ***)(a1 + 248);
        goto LABEL_9;
      }
      v18 = sub_15A4A70((__int64 ***)v10, v16);
      v17 = *(__int64 ***)(a1 + 248);
      v10 = v18;
    }
    v19 = v74;
LABEL_9:
    v65 = (int)v19;
    v74[0] = v10;
    v74[1] = a4;
    v20 = sub_15A0680((__int64)v17, v14, 0);
    v21 = *(_QWORD *)(a1 + 376);
    v75 = v20;
    return (_QWORD *)sub_1285290(a2, *(_QWORD *)(v21 + 24), v21, v65, 3, (__int64)v72, 0);
  }
  v23 = (_QWORD *)a2[3];
  v64 = v14 >> *(_DWORD *)(a1 + 224);
  v73 = 257;
  v24 = sub_16471D0(v23, 0);
  v25 = *(__int64 ***)v10;
  v71 = 257;
  v26 = *(_QWORD *)(a1 + 248);
  v66 = (__int64 **)v24;
  if ( v25 != (__int64 **)v26 )
  {
    if ( *(_BYTE *)(v10 + 16) > 0x10u )
    {
      LOWORD(v75) = 257;
      v47 = sub_15FDFF0(v10, v26, (__int64)v74, 0);
      v48 = a2[1];
      v10 = v47;
      if ( v48 )
      {
        v63 = (__int64 *)a2[2];
        sub_157E9D0(v48 + 40, v47);
        v49 = *v63;
        v50 = *(_QWORD *)(v10 + 24) & 7LL;
        *(_QWORD *)(v10 + 32) = v63;
        v49 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v49 | v50;
        *(_QWORD *)(v49 + 8) = v10 + 24;
        *v63 = *v63 & 7 | (v10 + 24);
      }
      sub_164B780(v10, v70);
      v51 = *a2;
      if ( *a2 )
      {
        v69 = *a2;
        sub_1623A60((__int64)&v69, v51, 2);
        v52 = *(_QWORD *)(v10 + 48);
        v53 = v10 + 48;
        if ( v52 )
        {
          sub_161E7C0(v10 + 48, v52);
          v53 = v10 + 48;
        }
        v54 = (unsigned __int8 *)v69;
        *(_QWORD *)(v10 + 48) = v69;
        if ( v54 )
          sub_1623210((__int64)&v69, v54, v53);
      }
    }
    else
    {
      v10 = sub_15A4A70((__int64 ***)v10, v26);
    }
  }
  v27 = sub_182C3C0(a1, v10, (__int64)v25, a2, a5, a6, a7);
  v28 = (_QWORD *)v27;
  if ( v66 != *(__int64 ***)v27 )
  {
    if ( *(_BYTE *)(v27 + 16) > 0x10u )
    {
      LOWORD(v75) = 257;
      v55 = sub_15FDBD0(46, v27, (__int64)v66, (__int64)v74, 0);
      v56 = a2[1];
      v28 = (_QWORD *)v55;
      if ( v56 )
      {
        v57 = (unsigned __int64 *)a2[2];
        sub_157E9D0(v56 + 40, v55);
        v58 = v28[3];
        v59 = *v57;
        v28[4] = v57;
        v59 &= 0xFFFFFFFFFFFFFFF8LL;
        v28[3] = v59 | v58 & 7;
        *(_QWORD *)(v59 + 8) = v28 + 3;
        *v57 = *v57 & 7 | (unsigned __int64)(v28 + 3);
      }
      sub_164B780((__int64)v28, v72);
      v60 = *a2;
      if ( *a2 )
      {
        v69 = *a2;
        sub_1623A60((__int64)&v69, v60, 2);
        v61 = v28[6];
        if ( v61 )
          sub_161E7C0((__int64)(v28 + 6), v61);
        v62 = (unsigned __int8 *)v69;
        v28[6] = v69;
        if ( v62 )
          sub_1623210((__int64)&v69, v62, (__int64)(v28 + 6));
      }
    }
    else
    {
      v28 = (_QWORD *)sub_15A46C0(46, (__int64 ***)v27, v66, 0);
    }
  }
  v29 = sub_1643360((_QWORD *)a2[3]);
  v30 = (__int64 *)sub_159C470(v29, v64, 0);
  return sub_15E7280(a2, v28, a4, v30, 1u, 0, 0, 0, 0);
}
