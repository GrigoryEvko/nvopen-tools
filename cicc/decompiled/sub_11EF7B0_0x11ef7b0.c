// Function: sub_11EF7B0
// Address: 0x11ef7b0
//
__int64 __fastcall sub_11EF7B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // r13
  size_t v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  _BYTE *v17; // rax
  unsigned int v18; // eax
  unsigned __int8 *v19; // rax
  __int64 v20; // rax
  bool v21; // al
  void **v22; // rax
  __int64 v23; // r8
  char v24; // dl
  void **v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  char v28; // r12
  unsigned int i; // ebx
  void **v30; // rax
  void **v31; // rdx
  char v32; // al
  _BYTE *v33; // rdx
  char v34; // dl
  _BYTE *v35; // rax
  __int64 v36; // rdx
  int v37; // eax
  const void *v38; // rsi
  __int64 v39; // rax
  unsigned __int8 v40; // al
  unsigned __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdx
  _BYTE *v44; // rax
  __int64 v45; // rsi
  unsigned int v46; // edx
  int v47; // edi
  unsigned int j; // ecx
  unsigned int *v49; // rax
  int v50; // eax
  int v51; // eax
  unsigned int v52; // ecx
  char v53; // al
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  int v57; // [rsp+0h] [rbp-120h]
  _BYTE *v58; // [rsp+0h] [rbp-120h]
  _DWORD *v59; // [rsp+8h] [rbp-118h]
  __int64 v60; // [rsp+8h] [rbp-118h]
  void *v61; // [rsp+8h] [rbp-118h]
  const char *s1; // [rsp+10h] [rbp-110h]
  __int64 *v63; // [rsp+18h] [rbp-108h]
  size_t n; // [rsp+20h] [rbp-100h]
  char v65; // [rsp+28h] [rbp-F8h]
  _DWORD *v66; // [rsp+30h] [rbp-F0h]
  char v67; // [rsp+30h] [rbp-F0h]
  __int64 v68; // [rsp+30h] [rbp-F0h]
  char v69; // [rsp+38h] [rbp-E8h]
  _BYTE *v70; // [rsp+38h] [rbp-E8h]
  __int64 v71; // [rsp+38h] [rbp-E8h]
  void **v72; // [rsp+38h] [rbp-E8h]
  void **v73; // [rsp+38h] [rbp-E8h]
  void **v74; // [rsp+38h] [rbp-E8h]
  __int64 *v75; // [rsp+38h] [rbp-E8h]
  __int64 v77; // [rsp+48h] [rbp-D8h]
  int v78; // [rsp+50h] [rbp-D0h]
  __int16 v79; // [rsp+54h] [rbp-CCh]
  char v80; // [rsp+57h] [rbp-C9h]
  __int64 v81; // [rsp+58h] [rbp-C8h]
  __int64 v82; // [rsp+58h] [rbp-C8h]
  char v83; // [rsp+67h] [rbp-B9h] BYREF
  unsigned int v84; // [rsp+68h] [rbp-B8h]
  int v85; // [rsp+6Ch] [rbp-B4h]
  __int64 v86; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v87; // [rsp+78h] [rbp-A8h]
  char v88; // [rsp+7Ch] [rbp-A4h]
  void *v89[4]; // [rsp+80h] [rbp-A0h] BYREF
  void *v90[4]; // [rsp+A0h] [rbp-80h] BYREF
  void *v91[2]; // [rsp+C0h] [rbp-60h] BYREF
  char v92; // [rsp+D4h] [rbp-4Ch]
  __int16 v93; // [rsp+E0h] [rbp-40h]

  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v7 = *(_QWORD *)(a2 - 32 * v6);
  v8 = *(_QWORD *)(a2 + 32 * (1 - v6));
  if ( v5 )
  {
    if ( *(_BYTE *)v5 )
    {
      v5 = 0;
    }
    else if ( *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v5 = 0;
    }
  }
  s1 = sub_BD5D20(v5);
  n = v9;
  v81 = *(_QWORD *)(a2 + 8);
  v63 = (__int64 *)sub_B43CA0(a2);
  v65 = sub_B45200(a2);
  v78 = *(_DWORD *)(a3 + 104);
  v77 = *(_QWORD *)(a3 + 96);
  v80 = *(_BYTE *)(a3 + 110);
  v79 = *(_WORD *)(a3 + 108);
  *(_DWORD *)(a3 + 104) = sub_B45210(a2);
  if ( *(_BYTE *)v7 == 18 )
  {
    v66 = sub_C33320();
    sub_C3B1B0((__int64)v91, 1.0);
    sub_C407B0(v90, (__int64 *)v91, v66);
    sub_C338F0((__int64)v91);
    sub_C41640((__int64 *)v90, *(_DWORD **)(v7 + 24), 1, (bool *)v91);
    v13 = v7;
  }
  else
  {
    v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v15 > 1 )
      goto LABEL_11;
    if ( *(_BYTE *)v7 > 0x15u )
      goto LABEL_11;
    v17 = sub_AD7630(v7, 0, v15);
    if ( !v17 )
      goto LABEL_11;
    v70 = v17;
    if ( *v17 != 18 )
      goto LABEL_11;
    v59 = sub_C33320();
    sub_C3B1B0((__int64)v91, 1.0);
    sub_C407B0(v90, (__int64 *)v91, v59);
    sub_C338F0((__int64)v91);
    sub_C41640((__int64 *)v90, *((_DWORD **)v70 + 3), 1, (bool *)v91);
    v12 = (__int64)v70;
    v13 = (__int64)v70;
  }
  v69 = sub_AC3090(v13, v90, v10, v11, v12);
  sub_91D830(v90);
  if ( v69 )
    goto LABEL_8;
LABEL_11:
  v16 = sub_11EAE00(a1, a2, (unsigned int **)a3);
  if ( v16 )
  {
    v7 = v16;
    goto LABEL_8;
  }
  v91[0] = (void *)0xBFF0000000000000LL;
  v18 = sub_1009690((double *)v91, v8);
  if ( (_BYTE)v18 )
  {
    v91[0] = "reciprocal";
    v93 = 259;
    v19 = sub_AD8DD0(v81, 1.0);
    HIDWORD(v90[0]) = 0;
    v7 = sub_A82920((unsigned int **)a3, v19, (_BYTE *)v7, LODWORD(v90[0]), (__int64)v91, 0);
    goto LABEL_8;
  }
  if ( *(_BYTE *)v8 == 18 )
  {
    if ( *(void **)(v8 + 24) == sub_C33340() )
      v20 = *(_QWORD *)(v8 + 32);
    else
      v20 = v8 + 24;
    v21 = (*(_BYTE *)(v20 + 20) & 7) == 3;
  }
  else
  {
    v67 = v18;
    v71 = *(_QWORD *)(v8 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v71 + 8) - 17 > 1 || *(_BYTE *)v8 > 0x15u )
      goto LABEL_33;
    v22 = (void **)sub_AD7630(v8, 0, v18);
    v23 = v71;
    v24 = v67;
    if ( !v22 || (v72 = v22, *(_BYTE *)v22 != 18) )
    {
      if ( *(_BYTE *)(v23 + 8) == 17 )
      {
        v57 = *(_DWORD *)(v23 + 32);
        if ( v57 )
        {
          v68 = v7;
          v28 = v24;
          v60 = a3;
          for ( i = 0; i != v57; ++i )
          {
            v30 = (void **)sub_AD69F0((unsigned __int8 *)v8, i);
            v31 = v30;
            if ( !v30 )
            {
LABEL_32:
              v7 = v68;
              a3 = v60;
              goto LABEL_33;
            }
            v32 = *(_BYTE *)v30;
            v73 = v31;
            if ( v32 != 13 )
            {
              if ( v32 != 18 )
                goto LABEL_32;
              v33 = v31[3] == sub_C33340() ? v73[4] : v73 + 3;
              if ( (v33[20] & 7) != 3 )
                goto LABEL_32;
              v28 = 1;
            }
          }
          v34 = v28;
          a3 = v60;
          v7 = v68;
          if ( v34 )
            goto LABEL_24;
        }
      }
      goto LABEL_33;
    }
    if ( v22[3] == sub_C33340() )
      v25 = (void **)v72[4];
    else
      v25 = v72 + 3;
    v21 = (*((_BYTE *)v25 + 20) & 7) == 3;
  }
  if ( v21 )
  {
LABEL_24:
    v7 = (__int64)sub_AD8DD0(v81, 1.0);
    goto LABEL_8;
  }
LABEL_33:
  v91[0] = (void *)0x3FF0000000000000LL;
  if ( (unsigned __int8)sub_1009690((double *)v91, v8) )
    goto LABEL_8;
  v91[0] = (void *)0x4000000000000000LL;
  if ( (unsigned __int8)sub_1009690((double *)v91, v8) )
  {
    HIDWORD(v90[0]) = 0;
    v93 = 259;
    v91[0] = "square";
    v7 = sub_A826E0((unsigned int **)a3, (_BYTE *)v7, (_BYTE *)v7, LODWORD(v90[0]), (__int64)v91, 0);
    goto LABEL_8;
  }
  v82 = sub_11E46E0((_QWORD *)a1, a2, (unsigned int **)a3);
  if ( v82 )
  {
    v7 = v82;
    goto LABEL_8;
  }
  if ( !v65 )
  {
LABEL_53:
    if ( *(_BYTE *)(a1 + 80) )
    {
      v35 = *(_BYTE **)(a1 + 24);
      if ( (v35[56] & 4) != 0
        || (v36 = *(_QWORD *)v35, (v37 = ((int)*(unsigned __int8 *)(*(_QWORD *)v35 + 96LL) >> 4) & 3) == 0) )
      {
        if ( n )
          goto LABEL_64;
      }
      else
      {
        if ( v37 == 3 )
        {
          v38 = "amd_vrd2_pow" + 9;
          v39 = qword_4977328[772];
        }
        else
        {
          v45 = *(_QWORD *)(v36 + 144);
          v46 = *(_DWORD *)(v36 + 160);
          if ( v46 )
          {
            v26 = v46 - 1;
            v47 = 1;
            for ( j = ((_WORD)v46 - 1) & 0x37CA; ; j = v26 & v52 )
            {
              v49 = (unsigned int *)(v45 + 40LL * j);
              v27 = *v49;
              if ( (_DWORD)v27 == 386 )
                break;
              v27 = (unsigned int)(v27 + 1);
              if ( !(_DWORD)v27 )
                goto LABEL_101;
              v52 = v47 + j;
              ++v47;
            }
          }
          else
          {
LABEL_101:
            v49 = (unsigned int *)(v45 + 40LL * v46);
          }
          v38 = (const void *)*((_QWORD *)v49 + 1);
          v39 = *((_QWORD *)v49 + 2);
        }
        if ( v39 != n || n && memcmp(s1, v38, n) )
          goto LABEL_64;
      }
      if ( (unsigned __int8)sub_11E9B60(a1, v63, (__int64)s1, n, v26, v27) )
      {
        v7 = sub_11DB650(a2, a3, 1, *(__int64 **)(a1 + 24), 1);
        goto LABEL_8;
      }
    }
LABEL_64:
    v7 = 0;
    goto LABEL_8;
  }
  v40 = *(_BYTE *)v8;
  if ( *(_BYTE *)v8 == 18 )
  {
    v74 = (void **)(v8 + 24);
    goto LABEL_69;
  }
  v43 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
  if ( (unsigned int)v43 <= 1 && v40 <= 0x15u )
  {
    v44 = sub_AD7630(v8, 0, v43);
    if ( !v44 || *v44 != 18 )
      goto LABEL_70;
    v74 = (void **)(v44 + 24);
LABEL_69:
    if ( (unsigned __int8)sub_11DB340((_DWORD **)v74, 0.5) || (unsigned __int8)sub_11DB340((_DWORD **)v74, -0.5) )
      goto LABEL_70;
    sub_9693D0((__int64)v91, v74);
    v61 = sub_C33340();
    if ( v91[0] == v61 )
    {
      if ( (*((_BYTE *)v91[1] + 20) & 8) != 0 )
        sub_C3CCB0((__int64)v91);
    }
    else if ( (v92 & 8) != 0 )
    {
      sub_C34440((unsigned __int8 *)v91);
    }
    if ( v91[0] == v61 )
      sub_C3C840(v89, v91);
    else
      sub_C338E0((__int64)v89, (__int64)v91);
    sub_91D830(v91);
    sub_9693D0((__int64)v90, v74);
    if ( (unsigned __int8)sub_11DB470(v89) )
    {
      v58 = 0;
    }
    else
    {
      sub_9693D0((__int64)v91, v89);
      if ( v61 == v91[0] )
        v50 = sub_C3D800((__int64 *)v91, (__int64)v89, 1u);
      else
        v50 = sub_C3ADF0((__int64)v91, (__int64)v89, 1);
      if ( v50
        || !(unsigned __int8)sub_11DB470(v91)
        || (v61 == v90[0] ? (v51 = sub_C3E740(v90, 3u)) : (v51 = sub_C3BAB0((__int64)v90, 3)),
            v51 != 16
         || !(unsigned __int8)sub_11DB470(v90)
         || (v75 = *(__int64 **)(a1 + 24),
             v53 = sub_B49E00(a2),
             (v58 = (_BYTE *)sub_11D9A40(v7, 0, v53, v63, a3, v75)) == 0)) )
      {
        sub_91D830(v91);
        goto LABEL_100;
      }
      sub_91D830(v91);
      v74 = v90;
    }
    v87 = *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL);
    if ( v87 > 0x40 )
      sub_C43690((__int64)&v86, 0, 0);
    else
      v86 = 0;
    v88 = 0;
    if ( !(unsigned __int8)sub_11DB470(v74) || (unsigned int)sub_C41980(v74, (__int64)&v86, 0, &v83) )
    {
      sub_969240(&v86);
      sub_91D830(v90);
      sub_91D830(v89);
LABEL_70:
      v40 = *(_BYTE *)v8;
      goto LABEL_71;
    }
    v54 = sub_BCD140(*(_QWORD **)(a3 + 72), *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL));
    v55 = sub_AD8D80(v54, (__int64)&v86);
    v56 = sub_11D99E0(v7, v55, a3);
    v82 = v56;
    if ( v56 )
    {
      if ( *(_BYTE *)v56 == 85 )
        *(_WORD *)(v56 + 2) = *(_WORD *)(v56 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
      if ( v58 )
      {
        v85 = 0;
        v93 = 257;
        v82 = sub_A826E0((unsigned int **)a3, (_BYTE *)v56, v58, v84, (__int64)v91, 0);
      }
    }
    sub_969240(&v86);
LABEL_100:
    sub_91D830(v90);
    sub_91D830(v89);
    v7 = v82;
    goto LABEL_8;
  }
LABEL_71:
  if ( (unsigned __int8)(v40 - 72) > 1u )
    goto LABEL_53;
  v41 = sub_11DBA30((char *)v8, a3, *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL));
  if ( !v41 )
    goto LABEL_53;
  v42 = sub_11D99E0(v7, v41, a3);
  v7 = v42;
  if ( !v42 )
    goto LABEL_64;
  if ( *(_BYTE *)v42 == 85 )
    *(_WORD *)(v42 + 2) = *(_WORD *)(v42 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
LABEL_8:
  *(_QWORD *)(a3 + 96) = v77;
  *(_DWORD *)(a3 + 104) = v78;
  *(_WORD *)(a3 + 108) = v79;
  *(_BYTE *)(a3 + 110) = v80;
  return v7;
}
