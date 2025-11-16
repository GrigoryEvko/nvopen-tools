// Function: sub_2D5EFD0
// Address: 0x2d5efd0
//
__int64 __fastcall sub_2D5EFD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, unsigned int a6)
{
  __int64 v9; // rdi
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // r9
  int v13; // edx
  unsigned int v14; // esi
  __int64 *v15; // rax
  __int64 v16; // r11
  __int64 v17; // r15
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 *v20; // rax
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 v24; // r13
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // r14
  __int64 (__fastcall *v34)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  unsigned int *v42; // r15
  unsigned int *v43; // r14
  __int64 v44; // rdx
  unsigned int v45; // esi
  int v46; // eax
  int v47; // eax
  int v48; // r8d
  int v49; // r10d
  __int64 v50; // [rsp+0h] [rbp-160h]
  __int64 v51; // [rsp+18h] [rbp-148h]
  __int64 *v53; // [rsp+20h] [rbp-140h]
  unsigned int v55; // [rsp+3Ch] [rbp-124h] BYREF
  _QWORD v56[4]; // [rsp+40h] [rbp-120h] BYREF
  char v57; // [rsp+60h] [rbp-100h]
  char v58; // [rsp+61h] [rbp-FFh]
  _QWORD v59[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v60; // [rsp+90h] [rbp-D0h]
  unsigned int *v61; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+A8h] [rbp-B8h]
  _BYTE v63[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v64; // [rsp+D0h] [rbp-90h]
  __int64 v65; // [rsp+D8h] [rbp-88h]
  __int64 v66; // [rsp+E0h] [rbp-80h]
  __int64 v67; // [rsp+E8h] [rbp-78h]
  void **v68; // [rsp+F0h] [rbp-70h]
  void **v69; // [rsp+F8h] [rbp-68h]
  __int64 v70; // [rsp+100h] [rbp-60h]
  int v71; // [rsp+108h] [rbp-58h]
  __int16 v72; // [rsp+10Ch] [rbp-54h]
  char v73; // [rsp+10Eh] [rbp-52h]
  __int64 v74; // [rsp+110h] [rbp-50h]
  __int64 v75; // [rsp+118h] [rbp-48h]
  void *v76; // [rsp+120h] [rbp-40h] BYREF
  void *v77; // [rsp+128h] [rbp-38h] BYREF

  if ( *(_QWORD *)(a2 + 40) != a5[5] )
  {
    v51 = a4;
    if ( !sub_2D59C50((char *)a2, *(_QWORD *)(a1 + 56)) )
      return 0;
    v9 = *(_QWORD *)(a2 + 40);
    v10 = *(_QWORD *)(a1 + 56);
    v11 = *(_DWORD *)(v10 + 24);
    v12 = *(_QWORD *)(v10 + 8);
    if ( v11 )
    {
      v13 = v11 - 1;
      v14 = v13 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v9 == *v15 )
      {
LABEL_5:
        v17 = v15[1];
      }
      else
      {
        v46 = 1;
        while ( v16 != -4096 )
        {
          v48 = v46 + 1;
          v14 = v13 & (v46 + v14);
          v15 = (__int64 *)(v12 + 16LL * v14);
          v16 = *v15;
          if ( v9 == *v15 )
            goto LABEL_5;
          v46 = v48;
        }
        v17 = 0;
      }
      v18 = a5[5];
      v19 = v13 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v20 = (__int64 *)(v12 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == v18 )
      {
LABEL_7:
        v22 = v20[1];
      }
      else
      {
        v47 = 1;
        while ( v21 != -4096 )
        {
          v49 = v47 + 1;
          v19 = v13 & (v47 + v19);
          v20 = (__int64 *)(v12 + 16LL * v19);
          v21 = *v20;
          if ( *v20 == v18 )
            goto LABEL_7;
          v47 = v49;
        }
        v22 = 0;
      }
      if ( v22 != v17 )
        return 0;
    }
    else
    {
      v17 = 0;
    }
    v24 = sub_2D57F20(a1, *(_QWORD *)(v9 + 72));
    v25 = sub_B19720(v24, a5[5], *(_QWORD *)(a2 + 40));
    a4 = v51;
    if ( !v25 )
    {
      v26 = *(_QWORD *)(a2 + 16);
      if ( !v26 )
        return 0;
      if ( *(_QWORD *)(v26 + 8) )
        return 0;
      v27 = sub_D47930(v17);
      v28 = sub_B19720(v24, a5[5], v27);
      a4 = v51;
      if ( !v28 )
        return 0;
    }
  }
  if ( *(_BYTE *)a2 == 42 && a6 == 372 )
    a4 = sub_AD6890(a4, 0);
  v29 = a5[5];
  v30 = *(_QWORD *)(v29 + 56);
  v31 = v29 + 48;
  if ( v30 == v31 )
  {
LABEL_39:
    v32 = 0;
  }
  else
  {
    while ( 1 )
    {
      if ( v30 )
      {
        v32 = v30 - 24;
        if ( *(_BYTE *)a2 != 59 && a2 == v32 )
          break;
        if ( (_QWORD *)v32 == a5 )
          break;
      }
      v30 = *(_QWORD *)(v30 + 8);
      if ( v31 == v30 )
        goto LABEL_39;
    }
  }
  v50 = a4;
  v67 = sub_BD5C60(v32);
  v68 = &v76;
  v69 = &v77;
  v61 = (unsigned int *)v63;
  v76 = &unk_49DA100;
  v62 = 0x200000000LL;
  v72 = 512;
  LOWORD(v66) = 0;
  v77 = &unk_49DA0B0;
  v70 = 0;
  v71 = 0;
  v73 = 7;
  v74 = 0;
  v75 = 0;
  v64 = 0;
  v65 = 0;
  sub_D5F1F0((__int64)&v61, v32);
  v60 = 257;
  HIDWORD(v56[0]) = 0;
  v33 = sub_B33C40((__int64)&v61, a6, a3, v50, LODWORD(v56[0]), (__int64)v59);
  v53 = (__int64 *)(a1 + 840);
  if ( *(_BYTE *)a2 != 59 )
  {
    v59[0] = "math";
    v60 = 259;
    LODWORD(v56[0]) = 0;
    v38 = sub_94D3D0(&v61, v33, (__int64)v56, 1, (__int64)v59);
    sub_2D594F0(a2, v38, v53, *(unsigned __int8 *)(a1 + 832), v39, v40);
  }
  v58 = 1;
  v56[0] = "ov";
  v57 = 3;
  v55 = 1;
  v34 = (__int64 (__fastcall *)(__int64, _BYTE *, __int64, __int64))*((_QWORD *)*v68 + 10);
  if ( v34 == sub_92FAE0 )
  {
    if ( *(_BYTE *)v33 > 0x15u )
    {
LABEL_34:
      v60 = 257;
      v37 = (__int64)sub_BD2C40(104, 1u);
      if ( v37 )
      {
        v41 = sub_B501B0(*(_QWORD *)(v33 + 8), &v55, 1);
        sub_B44260(v37, v41, 64, 1u, 0, 0);
        sub_AC2B30(v37 - 32, v33);
        *(_QWORD *)(v37 + 72) = v37 + 88;
        *(_QWORD *)(v37 + 80) = 0x400000000LL;
        sub_B50030(v37, &v55, 1, (__int64)v59);
      }
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v69 + 2))(v69, v37, v56, v65, v66);
      v42 = v61;
      v43 = &v61[4 * (unsigned int)v62];
      if ( v61 != v43 )
      {
        do
        {
          v44 = *((_QWORD *)v42 + 1);
          v45 = *v42;
          v42 += 4;
          sub_B99FD0(v37, v45, v44);
        }
        while ( v43 != v42 );
      }
      goto LABEL_31;
    }
    v37 = sub_AAADB0(v33, &v55, 1);
  }
  else
  {
    v37 = v34((__int64)v68, (_BYTE *)v33, (__int64)&v55, 1);
  }
  if ( !v37 )
    goto LABEL_34;
LABEL_31:
  sub_2D594F0((__int64)a5, v37, v53, *(unsigned __int8 *)(a1 + 832), v35, v36);
  sub_B43D60(a5);
  sub_B43D60((_QWORD *)a2);
  nullsub_61();
  v76 = &unk_49DA100;
  nullsub_63();
  if ( v61 != (unsigned int *)v63 )
    _libc_free((unsigned __int64)v61);
  return 1;
}
