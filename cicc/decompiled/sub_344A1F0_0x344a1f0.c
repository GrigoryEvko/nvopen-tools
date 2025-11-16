// Function: sub_344A1F0
// Address: 0x344a1f0
//
__int64 __fastcall sub_344A1F0(
        unsigned int *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  unsigned __int64 v12; // r14
  unsigned int v13; // r12d
  __int64 v14; // rbx
  __int64 v15; // rdi
  int v16; // edx
  bool v17; // zf
  __int64 v18; // r9
  int v19; // r11d
  __int64 v20; // r13
  __int64 v21; // rcx
  __int16 v22; // ax
  __int64 v23; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  bool v27; // al
  __int64 v28; // rsi
  unsigned int v29; // eax
  unsigned int v30; // eax
  bool v31; // al
  __int64 v32; // rax
  int v33; // ecx
  __int64 v34; // rdx
  __int64 (*v35)(); // rax
  __int64 v36; // rax
  int v37; // edx
  bool v38; // al
  __int64 v39; // rax
  unsigned int v40; // esi
  __int64 v41; // r8
  __int64 v42; // rax
  char v43; // al
  int v44; // eax
  __int64 (*v45)(); // rax
  __int64 v46; // rdx
  __int64 v47; // r9
  __int64 v48; // rdx
  unsigned __int64 v49; // rdx
  int v50; // eax
  unsigned int v51; // eax
  __int64 v52; // rdx
  __int64 (*v53)(); // rax
  char v54; // al
  unsigned __int8 *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r13
  __int64 v58; // r12
  __int128 v59; // rax
  __int64 v60; // [rsp+0h] [rbp-B0h]
  __int64 v61; // [rsp+0h] [rbp-B0h]
  __int64 v62; // [rsp+0h] [rbp-B0h]
  int v63; // [rsp+8h] [rbp-A8h]
  __int64 v64; // [rsp+8h] [rbp-A8h]
  __int64 v65; // [rsp+8h] [rbp-A8h]
  int v66; // [rsp+8h] [rbp-A8h]
  __int64 v67; // [rsp+10h] [rbp-A0h]
  __int64 v68; // [rsp+10h] [rbp-A0h]
  __int128 v69; // [rsp+10h] [rbp-A0h]
  int v70; // [rsp+10h] [rbp-A0h]
  int v71; // [rsp+10h] [rbp-A0h]
  __int64 v72; // [rsp+10h] [rbp-A0h]
  int v73; // [rsp+20h] [rbp-90h]
  __int128 v74; // [rsp+20h] [rbp-90h]
  __int64 v75; // [rsp+20h] [rbp-90h]
  __int64 v76; // [rsp+20h] [rbp-90h]
  __int64 v77; // [rsp+20h] [rbp-90h]
  unsigned int v78; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v81; // [rsp+38h] [rbp-78h]
  __int128 v82; // [rsp+40h] [rbp-70h]
  __int64 v83; // [rsp+50h] [rbp-60h] BYREF
  __int64 v84; // [rsp+58h] [rbp-58h]
  __int128 v85; // [rsp+60h] [rbp-50h] BYREF
  __int64 v86[8]; // [rsp+70h] [rbp-40h] BYREF

  v12 = a6;
  v13 = a5;
  v14 = a4;
  v15 = *(_QWORD *)(a11 + 16);
  v16 = *(_DWORD *)(a4 + 24);
  v83 = a4;
  v17 = *(_DWORD *)(a8 + 24) == 186;
  v84 = a5;
  v18 = a8;
  v19 = a9;
  if ( v17 )
  {
    if ( v16 != 186 )
    {
      v83 = a8;
      v19 = a5;
      v13 = a9;
      v18 = a4;
      LODWORD(v84) = a9;
      v14 = a8;
    }
    v20 = 16LL * v13;
    v25 = v20 + *(_QWORD *)(v14 + 48);
    v22 = *(_WORD *)v25;
    v26 = *(_QWORD *)(v25 + 8);
    LOWORD(v85) = v22;
    *((_QWORD *)&v85 + 1) = v26;
  }
  else
  {
    v20 = 16LL * (unsigned int)a5;
    v21 = v20 + *(_QWORD *)(a4 + 48);
    v22 = *(_WORD *)v21;
    v23 = *(_QWORD *)(v21 + 8);
    LOWORD(v85) = v22;
    *((_QWORD *)&v85 + 1) = v23;
    if ( v16 != 186 )
      return 0;
  }
  if ( v22 )
  {
    if ( (unsigned __int16)(v22 - 2) > 7u
      && (unsigned __int16)(v22 - 17) > 0x6Cu
      && (unsigned __int16)(v22 - 176) > 0x1Fu )
    {
      return 0;
    }
  }
  else
  {
    v68 = v18;
    v73 = v19;
    v31 = sub_3007070((__int64)&v85);
    v19 = v73;
    v18 = v68;
    if ( !v31 )
      return 0;
  }
  if ( (_DWORD)v12 != 17 )
  {
    if ( (_DWORD)v12 != 22 )
      return 0;
    v63 = v19;
    v67 = v18;
    v27 = sub_33CF170(v18);
    v18 = v67;
    v19 = v63;
    if ( v27 )
    {
      v28 = (unsigned int)v85;
      v29 = sub_3289F80(a1, (unsigned int)v85, *((__int64 *)&v85 + 1));
      v19 = v63;
      v18 = v67;
      if ( v29 <= 1 )
      {
        v30 = sub_32844A0((unsigned __int16 *)&v85, v28);
        sub_109DDE0((__int64)v86, v30, v30 - 1);
        if ( (unsigned __int8)sub_33DD210(v15, v83, v84, (__int64)v86, 0) )
        {
          v81 = sub_33FB620(v15, v83, v84, a10, a2, a3, a7, v85);
          sub_969240(v86);
          return (__int64)v81;
        }
        sub_969240(v86);
        v18 = v67;
        v19 = v63;
      }
    }
  }
  v32 = *(_QWORD *)(v14 + 40);
  v33 = *(_DWORD *)(*(_QWORD *)(v32 + 40) + 24LL);
  if ( v33 != 11 && v33 != 35 )
    goto LABEL_21;
  v65 = *(_QWORD *)(v32 + 40);
  v70 = v19;
  v75 = v18;
  v38 = sub_33CF170(v18);
  v18 = v75;
  v19 = v70;
  if ( v38 )
  {
    v39 = *(_QWORD *)(v65 + 96);
    v40 = *(_DWORD *)(v39 + 32);
    v41 = v39 + 24;
    if ( v40 > 0x40 )
    {
      v62 = v75;
      v77 = v39 + 24;
      v44 = sub_C44630(v39 + 24);
      v41 = v77;
      v19 = v70;
      v18 = v62;
      if ( v44 == 1 )
        goto LABEL_39;
    }
    else
    {
      v42 = *(_QWORD *)(v39 + 24);
      if ( v42 && (v42 & (v42 - 1)) == 0 )
      {
LABEL_39:
        if ( (_WORD)v85 )
        {
          if ( *(_QWORD *)&a1[2 * (unsigned __int16)v85 + 28] )
          {
            v61 = v18;
            v71 = v19;
            v76 = v41;
            v43 = sub_3286E00(&v83);
            v19 = v71;
            v18 = v61;
            if ( v43 )
            {
              v50 = sub_9871A0(v76);
              v51 = sub_327FC40(*(_QWORD **)(v15 + 64), v40 - v50);
              v19 = v71;
              v18 = v61;
              v78 = v51;
              v53 = *(__int64 (**)())(*(_QWORD *)a1 + 1392LL);
              if ( v53 != sub_2FE3480 )
              {
                v66 = v71;
                v72 = v52;
                v54 = ((__int64 (__fastcall *)(unsigned int *, _QWORD, _QWORD, _QWORD, __int64))v53)(
                        a1,
                        (unsigned int)v85,
                        *((_QWORD *)&v85 + 1),
                        v78,
                        v52);
                v19 = v66;
                v18 = v61;
                if ( v54 )
                {
                  if ( (_WORD)v78 && *(_QWORD *)&a1[2 * (unsigned __int16)v78 + 28] )
                  {
                    v55 = sub_33FB310(
                            v15,
                            **(_QWORD **)(v14 + 40),
                            *(_QWORD *)(*(_QWORD *)(v14 + 40) + 8LL),
                            a10,
                            v78,
                            v72,
                            a7);
                    v57 = v56;
                    v58 = (__int64)v55;
                    *(_QWORD *)&v59 = sub_3400BD0(v15, 0, a10, v78, v72, 0, a7, 0);
                    return sub_32889F0(v15, a10, a2, a3, v58, v57, v59, (unsigned int)((_DWORD)v12 != 17) + 19, 0);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v32 = *(_QWORD *)(v14 + 40);
LABEL_21:
  if ( *(_QWORD *)v32 == v18 && *(_DWORD *)(v32 + 8) == v19 )
  {
    *(_QWORD *)&v74 = v18;
    v64 = *(_QWORD *)(v32 + 40);
    v60 = *(unsigned int *)(v32 + 48);
    *((_QWORD *)&v74 + 1) = *(unsigned int *)(v32 + 8);
  }
  else
  {
    if ( *(_QWORD *)(v32 + 40) != v18 || *(_DWORD *)(v32 + 48) != v19 )
      return 0;
    *(_QWORD *)&v74 = v18;
    v64 = *(_QWORD *)v32;
    v60 = *(unsigned int *)(v32 + 8);
    *((_QWORD *)&v74 + 1) = *(unsigned int *)(v32 + 48);
  }
  *(_QWORD *)&v69 = sub_3400BD0(v15, 0, a10, (unsigned int)v85, *((__int64 *)&v85 + 1), 0, a7, 0);
  *((_QWORD *)&v69 + 1) = v34;
  v35 = *(__int64 (**)())(*(_QWORD *)a1 + 2648LL);
  if ( (v35 == sub_302E2C0
     || ((unsigned __int8 (__fastcall *)(unsigned int *, _QWORD, _QWORD, _QWORD))v35)(
          a1,
          (unsigned int)v12,
          (unsigned int)v85,
          *((_QWORD *)&v85 + 1)))
    && (unsigned __int8)sub_33E0A10(v15, v74, *((__int64 *)&v74 + 1), 0) )
  {
    v49 = (unsigned int)sub_33CBD40(v12, (unsigned int)v85, *((__int64 *)&v85 + 1));
    if ( *(int *)(a11 + 8) > 1
      && ((a1[((*(_WORD *)(*(_QWORD *)(v14 + 48) + v20) >> 3) & 0x1FFF) + 36LL * (int)v49 - (int)v49 + 130384] >> (4 * (*(_WORD *)(*(_QWORD *)(v14 + 48) + v20) & 7)))
        & 0xF) != 0 )
    {
      return 0;
    }
    return sub_32889F0(v15, a10, a2, a3, v83, v84, v69, v49, 0);
  }
  else
  {
    v36 = *(_QWORD *)(v14 + 56);
    if ( !v36 )
      return 0;
    v37 = 1;
    do
    {
      if ( v13 == *(_DWORD *)(v36 + 8) )
      {
        if ( !v37 )
          return 0;
        v36 = *(_QWORD *)(v36 + 32);
        if ( !v36 )
          goto LABEL_48;
        if ( v13 == *(_DWORD *)(v36 + 8) )
          return 0;
        v37 = 0;
      }
      v36 = *(_QWORD *)(v36 + 32);
    }
    while ( v36 );
    if ( v37 == 1 )
      return 0;
LABEL_48:
    v45 = *(__int64 (**)())(*(_QWORD *)a1 + 384LL);
    if ( v45 == sub_2FE3020
      || !((unsigned __int8 (__fastcall *)(unsigned int *, _QWORD, _QWORD))v45)(a1, v74, *((_QWORD *)&v74 + 1))
      || sub_33CF170(v74) )
    {
      return 0;
    }
    sub_3285E70((__int64)v86, v64);
    *(_QWORD *)&v82 = sub_34074A0((_QWORD *)v15, (__int64)v86, v64, v60, v85, *((__int64 *)&v85 + 1), a7);
    *((_QWORD *)&v82 + 1) = v46;
    sub_9C6650(v86);
    sub_3285E70((__int64)v86, v83);
    *(_QWORD *)&v82 = sub_3406EB0(
                        (_QWORD *)v15,
                        0xBAu,
                        (__int64)v86,
                        (unsigned int)v85,
                        *((__int64 *)&v85 + 1),
                        v47,
                        v82,
                        v74);
    *((_QWORD *)&v82 + 1) = v48;
    sub_9C6650(v86);
    return sub_32889F0(v15, a10, a2, a3, v82, *((__int64 *)&v82 + 1), v69, v12, 0);
  }
}
