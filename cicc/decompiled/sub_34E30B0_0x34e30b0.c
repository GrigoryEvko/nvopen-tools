// Function: sub_34E30B0
// Address: 0x34e30b0
//
__int64 __fastcall sub_34E30B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r15
  unsigned __int64 v6; // rsi
  unsigned int *v7; // rax
  int v8; // ecx
  unsigned int *v9; // rdx
  unsigned __int8 *v11; // r15
  __int64 v12; // rax
  __int64 *v13; // r11
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int8 *v19; // r11
  __int64 (__fastcall *v20)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64, __int64); // rax
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rsi
  __int64 *v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int *v29; // rbx
  unsigned int *i; // r15
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // [rsp-10h] [rbp-180h]
  __int64 *v35; // [rsp+0h] [rbp-170h]
  _BYTE *v36; // [rsp+8h] [rbp-168h]
  __int64 v37; // [rsp+10h] [rbp-160h]
  unsigned __int8 *v38; // [rsp+10h] [rbp-160h]
  __int64 v39; // [rsp+10h] [rbp-160h]
  __int64 v40; // [rsp+10h] [rbp-160h]
  unsigned __int8 *v41; // [rsp+10h] [rbp-160h]
  __int64 v42; // [rsp+30h] [rbp-140h]
  unsigned int v43; // [rsp+38h] [rbp-138h]
  _QWORD v44[2]; // [rsp+40h] [rbp-130h] BYREF
  _QWORD v45[4]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v46; // [rsp+70h] [rbp-100h]
  __int64 v47[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v48; // [rsp+A0h] [rbp-D0h]
  unsigned int *v49; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+B8h] [rbp-B8h]
  _BYTE v51[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+E0h] [rbp-90h]
  __int64 v53; // [rsp+E8h] [rbp-88h]
  __int64 v54; // [rsp+F0h] [rbp-80h]
  _QWORD *v55; // [rsp+F8h] [rbp-78h]
  void **v56; // [rsp+100h] [rbp-70h]
  void **v57; // [rsp+108h] [rbp-68h]
  __int64 v58; // [rsp+110h] [rbp-60h]
  int v59; // [rsp+118h] [rbp-58h]
  __int16 v60; // [rsp+11Ch] [rbp-54h]
  char v61; // [rsp+11Eh] [rbp-52h]
  __int64 v62; // [rsp+120h] [rbp-50h]
  __int64 v63; // [rsp+128h] [rbp-48h]
  void *v64; // [rsp+130h] [rbp-40h] BYREF
  void *v65; // [rsp+138h] [rbp-38h] BYREF

  v56 = &v64;
  v55 = (_QWORD *)sub_BD5C60(a1);
  v50 = 0x200000000LL;
  v64 = &unk_49DA100;
  v49 = (unsigned int *)v51;
  v57 = &v65;
  v65 = &unk_49DA0B0;
  v1 = *(_QWORD *)(a1 + 40);
  v58 = 0;
  v52 = v1;
  v59 = 0;
  v60 = 512;
  v61 = 7;
  v62 = 0;
  v63 = 0;
  v53 = a1 + 24;
  LOWORD(v54) = 0;
  v2 = *(_QWORD *)sub_B46C60(a1);
  v47[0] = v2;
  if ( v2 && (sub_B96E90((__int64)v47, v2, 1), (v5 = v47[0]) != 0) )
  {
    v6 = (unsigned int)v50;
    v7 = v49;
    v8 = v50;
    v9 = &v49[4 * (unsigned int)v50];
    if ( v49 != v9 )
    {
      while ( *v7 )
      {
        v7 += 4;
        if ( v9 == v7 )
          goto LABEL_16;
      }
      *((_QWORD *)v7 + 1) = v47[0];
      goto LABEL_8;
    }
LABEL_16:
    if ( (unsigned int)v50 >= (unsigned __int64)HIDWORD(v50) )
    {
      v6 = (unsigned int)v50 + 1LL;
      if ( HIDWORD(v50) < v6 )
      {
        v6 = (unsigned __int64)v51;
        sub_C8D5F0((__int64)&v49, v51, (unsigned int)v50 + 1LL, 0x10u, v3, v4);
        v9 = &v49[4 * (unsigned int)v50];
      }
      *(_QWORD *)v9 = 0;
      *((_QWORD *)v9 + 1) = v5;
      v5 = v47[0];
      LODWORD(v50) = v50 + 1;
    }
    else
    {
      if ( v9 )
      {
        *v9 = 0;
        *((_QWORD *)v9 + 1) = v5;
        v8 = v50;
        v5 = v47[0];
      }
      LODWORD(v50) = v8 + 1;
    }
  }
  else
  {
    v6 = 0;
    sub_93FB40((__int64)&v49, 0);
    v5 = v47[0];
  }
  if ( v5 )
  {
LABEL_8:
    v6 = v5;
    sub_B91220((__int64)v47, v5);
  }
  if ( sub_B5AF00(a1, v6) )
    goto LABEL_10;
  v11 = (unsigned __int8 *)sub_B5A250(a1);
  v37 = sub_B5A450(a1);
  v12 = sub_B5A2C0(a1);
  v42 = v12;
  if ( BYTE4(v12) )
  {
    v24 = (__int64 *)sub_BCB2A0(v55);
    v25 = sub_BCE1B0(v24, v42);
    v26 = sub_BCB2D0(v55);
    v27 = sub_ACD640(v26, 0, 0);
    v45[0] = v25;
    v48 = 257;
    v44[1] = v37;
    v44[0] = v27;
    v45[1] = *(_QWORD *)(v37 + 8);
    v28 = sub_B33D10((__int64)&v49, 0xB9u, (__int64)v45, 2, (int)v44, 2, v43, (__int64)v47);
    v18 = v34;
    v19 = (unsigned __int8 *)v28;
  }
  else
  {
    v13 = *(__int64 **)(v37 + 8);
    v48 = 257;
    v35 = v13;
    v14 = sub_B37A60(&v49, v12, v37, v47);
    v48 = 257;
    v36 = (_BYTE *)v14;
    v15 = sub_BCE1B0(v35, v42);
    v16 = sub_B33FB0((__int64)&v49, v15, (__int64)v47);
    v48 = 257;
    v19 = (unsigned __int8 *)sub_92B530(&v49, 0x24u, v16, v36, (__int64)v47);
  }
  v46 = 257;
  v20 = (__int64 (__fastcall *)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64, __int64))*((_QWORD *)*v56 + 2);
  if ( (char *)v20 == (char *)sub_9202E0 )
  {
    if ( *v19 > 0x15u || *v11 > 0x15u )
    {
LABEL_34:
      v48 = 257;
      v39 = sub_B504D0(28, (__int64)v19, (__int64)v11, (__int64)v47, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v57 + 2))(v57, v39, v45, v53, v54);
      v22 = v39;
      v29 = &v49[4 * (unsigned int)v50];
      for ( i = v49; v29 != i; v22 = v40 )
      {
        v31 = *((_QWORD *)i + 1);
        v32 = *i;
        i += 4;
        v40 = v22;
        sub_B99FD0(v22, v32, v31);
      }
      goto LABEL_29;
    }
    v38 = v19;
    if ( (unsigned __int8)sub_AC47B0(28) )
      v21 = sub_AD5570(28, (__int64)v38, v11, 0, 0);
    else
      v21 = sub_AABE40(0x1Cu, v38, v11);
    v19 = v38;
    v22 = v21;
  }
  else
  {
    v41 = v19;
    v33 = v20(v56, 28, v19, v11, v17, v18);
    v19 = v41;
    v22 = v33;
  }
  if ( !v22 )
    goto LABEL_34;
LABEL_29:
  v23 = v22;
  sub_B5A320(a1, v22);
  if ( !sub_B5AF00(a1, v23) && sub_B5A450(a1) )
    sub_34E2BC0(a1);
LABEL_10:
  nullsub_61();
  v64 = &unk_49DA100;
  nullsub_63();
  if ( v49 != (unsigned int *)v51 )
    _libc_free((unsigned __int64)v49);
  return a1;
}
