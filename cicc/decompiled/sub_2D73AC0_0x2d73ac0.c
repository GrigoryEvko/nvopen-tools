// Function: sub_2D73AC0
// Address: 0x2d73ac0
//
void __fastcall sub_2D73AC0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // r14
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // r13
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int16 v13; // dx
  __int64 v14; // r8
  char v15; // si
  char v16; // al
  __int16 v17; // dx
  _QWORD *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rsi
  __int64 v21; // r9
  __int64 v22; // r8
  char *v23; // rax
  int v24; // ecx
  char *v25; // rdx
  __int64 v26; // rax
  _BYTE *v27; // r8
  __int64 *v28; // r12
  _BYTE *v29; // rax
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 *v34; // r12
  __int64 v35; // rdx
  unsigned __int64 v36; // rsi
  unsigned __int64 *v37; // rcx
  __int64 v38; // rdi
  unsigned __int64 *v39; // rbx
  unsigned __int64 *v40; // rax
  __int16 v41; // dx
  char v42; // si
  char v43; // al
  _QWORD *v44; // r15
  __int64 v45; // r12
  unsigned __int64 *v46; // r13
  _QWORD *v47; // rdx
  _QWORD *v48; // rsi
  __int64 v49; // rdi
  _QWORD *v50; // r12
  _QWORD *v51; // rbx
  __int16 v52; // dx
  char v53; // si
  char v54; // al
  unsigned __int64 v55; // rsi
  __int64 *v56; // r15
  unsigned __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // [rsp-10h] [rbp-1A0h]
  __int16 v60; // [rsp+8h] [rbp-188h]
  __int64 v61; // [rsp+10h] [rbp-180h]
  __int64 v62; // [rsp+10h] [rbp-180h]
  __int64 v64; // [rsp+28h] [rbp-168h]
  __int64 v65; // [rsp+28h] [rbp-168h]
  __int64 v66; // [rsp+30h] [rbp-160h]
  _BYTE *v67; // [rsp+30h] [rbp-160h]
  __int64 *v68; // [rsp+30h] [rbp-160h]
  __int64 v69; // [rsp+30h] [rbp-160h]
  _BYTE *v70; // [rsp+98h] [rbp-F8h] BYREF
  __int64 v71[2]; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v72; // [rsp+B0h] [rbp-E0h]
  __int16 v73; // [rsp+C0h] [rbp-D0h]
  char *v74; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+D8h] [rbp-B8h]
  _BYTE v76[16]; // [rsp+E0h] [rbp-B0h] BYREF
  __int16 v77; // [rsp+F0h] [rbp-A0h]
  __int64 v78; // [rsp+100h] [rbp-90h]
  __int64 v79; // [rsp+108h] [rbp-88h]
  __int16 v80; // [rsp+110h] [rbp-80h]
  _QWORD *v81; // [rsp+118h] [rbp-78h]
  void **v82; // [rsp+120h] [rbp-70h]
  void **v83; // [rsp+128h] [rbp-68h]
  __int64 v84; // [rsp+130h] [rbp-60h]
  int v85; // [rsp+138h] [rbp-58h]
  __int16 v86; // [rsp+13Ch] [rbp-54h]
  char v87; // [rsp+13Eh] [rbp-52h]
  __int64 v88; // [rsp+140h] [rbp-50h]
  __int64 v89; // [rsp+148h] [rbp-48h]
  void *v90; // [rsp+150h] [rbp-40h] BYREF
  void *v91; // [rsp+158h] [rbp-38h] BYREF

  v7 = (__int64 *)sub_BD5C60(a4);
  v66 = sub_AE4570((*a1)[102], *(_QWORD *)(a4 + 8));
  v8 = *(_QWORD *)(a4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  v64 = sub_BCE3C0(v7, *(_DWORD *)(v8 + 8) >> 8);
  v9 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x1Cu )
  {
    v10 = *(_QWORD *)(sub_B43CB0(*a1[1]) + 80);
    if ( v10 )
      v10 -= 24;
    v14 = sub_AA5190(v10);
    if ( v14 )
    {
      v42 = v41;
      v43 = HIBYTE(v41);
    }
    else
    {
      v43 = 0;
      v42 = 0;
    }
    LOBYTE(v17) = v42;
    HIBYTE(v17) = v43;
  }
  else
  {
    v10 = *(_QWORD *)(a3 + 40);
    if ( v9 == 84 )
    {
      v14 = sub_AA5190(*(_QWORD *)(a3 + 40));
      if ( v14 )
      {
        v53 = v52;
        v54 = HIBYTE(v52);
      }
      else
      {
        v54 = 0;
        v53 = 0;
      }
      LOBYTE(v17) = v53;
      HIBYTE(v17) = v54;
    }
    else if ( v9 == 34 )
    {
      v11 = *a1;
      v12 = *(_QWORD *)(a3 + 40);
      v77 = 257;
      v10 = sub_F41C30(v12, *(_QWORD *)(a3 - 96), v11[103], v11[7], 0, (void **)&v74);
      v14 = sub_AA5190(v10);
      if ( v14 )
      {
        v15 = v13;
        v16 = HIBYTE(v13);
      }
      else
      {
        v16 = 0;
        v15 = 0;
      }
      LOBYTE(v17) = v15;
      HIBYTE(v17) = v16;
    }
    else
    {
      v14 = *(_QWORD *)(a3 + 32);
      v17 = 0;
    }
  }
  v60 = v17;
  v61 = v14;
  v18 = (_QWORD *)sub_AA48A0(v10);
  v83 = &v91;
  v81 = v18;
  v82 = &v90;
  v86 = 512;
  v19 = v61;
  v78 = v10;
  v90 = &unk_49DA100;
  v74 = v76;
  v91 = &unk_49DA0B0;
  v75 = 0x200000000LL;
  v84 = 0;
  v85 = 0;
  v87 = 7;
  v88 = 0;
  v89 = 0;
  v79 = v61;
  v80 = v60;
  if ( v61 != v10 + 48 )
  {
    if ( v61 )
      v19 = v61 - 24;
    v20 = *(_QWORD *)sub_B46C60(v19);
    v71[0] = v20;
    if ( v20 && (sub_B96E90((__int64)v71, v20, 1), (v22 = v71[0]) != 0) )
    {
      v23 = v74;
      v24 = v75;
      v25 = &v74[16 * (unsigned int)v75];
      if ( v74 != v25 )
      {
        while ( *(_DWORD *)v23 )
        {
          v23 += 16;
          if ( v25 == v23 )
            goto LABEL_42;
        }
        *((_QWORD *)v23 + 1) = v71[0];
        goto LABEL_19;
      }
LABEL_42:
      if ( (unsigned int)v75 >= (unsigned __int64)HIDWORD(v75) )
      {
        v57 = (unsigned int)v75 + 1LL;
        if ( HIDWORD(v75) < v57 )
        {
          v62 = v71[0];
          sub_C8D5F0((__int64)&v74, v76, v57, 0x10u, v71[0], v21);
          v22 = v62;
          v25 = &v74[16 * (unsigned int)v75];
        }
        *(_QWORD *)v25 = 0;
        *((_QWORD *)v25 + 1) = v22;
        v22 = v71[0];
        LODWORD(v75) = v75 + 1;
      }
      else
      {
        if ( v25 )
        {
          *(_DWORD *)v25 = 0;
          *((_QWORD *)v25 + 1) = v22;
          v24 = v75;
          v22 = v71[0];
        }
        LODWORD(v75) = v24 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v74, 0);
      v22 = v71[0];
    }
    if ( v22 )
LABEL_19:
      sub_B91220((__int64)v71, v22);
  }
  v26 = sub_AD64C0(v66, a2, 0);
  v27 = (_BYTE *)v26;
  *a1[2] = a3;
  v28 = a1[2];
  if ( v64 != *(_QWORD *)(*v28 + 8) )
  {
    v67 = (_BYTE *)v26;
    v73 = 257;
    v29 = sub_94BCF0((unsigned int **)&v74, *v28, v64, (__int64)v71);
    v27 = v67;
    *v28 = (__int64)v29;
    v28 = a1[2];
  }
  v71[0] = (__int64)"splitgep";
  v73 = 259;
  v65 = *v28;
  v70 = v27;
  v30 = sub_BCB2B0(v81);
  *v28 = sub_921130((unsigned int **)&v74, v30, v65, &v70, 1, (__int64)v71, 0);
  v35 = v59;
  v33 = *a1[2];
  v34 = *a1;
  v71[0] = 0;
  v71[1] = 0;
  v72 = v33;
  LOBYTE(v35) = v33 != -4096;
  if ( ((unsigned __int8)v35 & (v33 != 0)) != 0 && v33 != -8192 )
    sub_BD73F0((__int64)v71);
  if ( v34[90] )
  {
    sub_2D705D0((__int64)(v34 + 85), v71);
    v35 = v72;
  }
  else
  {
    v36 = *((unsigned int *)v34 + 156);
    v37 = (unsigned __int64 *)v34[77];
    LODWORD(v38) = *((_DWORD *)v34 + 156);
    v39 = &v37[3 * v36];
    if ( v37 == v39 )
    {
      if ( v36 > 1 )
      {
        v44 = v34 + 85;
LABEL_60:
        *((_DWORD *)v34 + 156) = 0;
        sub_2D705D0((__int64)v44, v71);
        v35 = v72;
        goto LABEL_31;
      }
    }
    else
    {
      v35 = v72;
      v40 = (unsigned __int64 *)v34[77];
      while ( v40[2] != v72 )
      {
        v40 += 3;
        if ( v39 == v40 )
          goto LABEL_51;
      }
      if ( v40 != v39 )
        goto LABEL_31;
LABEL_51:
      if ( v36 > 1 )
      {
        v68 = v34;
        v44 = v34 + 85;
        v45 = (__int64)(v34 + 86);
        v46 = v37;
        do
        {
          v48 = sub_2D739C0(v44, v45, (__int64)v46);
          if ( v47 )
            sub_2D58620((__int64)v44, (__int64)v48, v47, v46);
          v46 += 3;
        }
        while ( v39 != v46 );
        v34 = v68;
        v49 = v68[77];
        if ( v49 + 24LL * *((unsigned int *)v68 + 156) != v49 )
        {
          v50 = (_QWORD *)(v49 + 24LL * *((unsigned int *)v68 + 156));
          v51 = (_QWORD *)v68[77];
          do
          {
            v50 -= 3;
            sub_D68D70(v50);
          }
          while ( v50 != v51 );
          v34 = v68;
        }
        goto LABEL_60;
      }
    }
    v55 = v36 + 1;
    v56 = v71;
    if ( v55 > *((unsigned int *)v34 + 157) )
    {
      v38 = (__int64)(v34 + 77);
      if ( v37 > (unsigned __int64 *)v71 || v39 <= (unsigned __int64 *)v71 )
      {
        sub_F39560(v38, v55, v35, (__int64)v37, v31, v32);
        LODWORD(v38) = *((_DWORD *)v34 + 156);
        v39 = (unsigned __int64 *)(v34[77] + 24LL * (unsigned int)v38);
      }
      else
      {
        v69 = v34[77];
        sub_F39560(v38, v55, v35, (__int64)v37, v31, v32);
        v58 = v34[77];
        v38 = *((unsigned int *)v34 + 156);
        v56 = (__int64 *)((char *)v71 + v58 - v69);
        v39 = (unsigned __int64 *)(v58 + 24 * v38);
      }
    }
    if ( v39 )
    {
      sub_D68CD0(v39, 0, v56);
      LODWORD(v38) = *((_DWORD *)v34 + 156);
    }
    *((_DWORD *)v34 + 156) = v38 + 1;
    v35 = v72;
  }
LABEL_31:
  if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
    sub_BD60C0(v71);
  nullsub_61();
  v90 = &unk_49DA100;
  nullsub_63();
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
}
