// Function: sub_2683090
// Address: 0x2683090
//
__int64 __fastcall sub_2683090(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v8; // r8d
  __int64 *v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // r14
  unsigned __int8 *v12; // r15
  int v13; // eax
  __int64 v14; // rsi
  __int64 v16; // rcx
  __int64 v17; // r14
  int v18; // ecx
  __int64 v19; // rcx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 *v22; // r13
  unsigned __int8 *v23; // rax
  unsigned __int8 *v24; // r14
  unsigned __int64 v25; // rax
  int v26; // edx
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int8 *v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r13
  int v40; // r13d
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rsi
  __int64 v44; // rdx
  unsigned __int8 *v45; // r10
  __int64 v46; // rax
  __int64 v47; // rax
  int v48; // edx
  int v49; // r11d
  __int64 v50; // rcx
  __int64 v51; // rsi
  int v52; // r13d
  unsigned __int64 v53; // rdx
  unsigned int i; // edx
  __int64 v55; // r8
  unsigned int v56; // edx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdx
  unsigned __int8 **v61; // rcx
  __int64 v62; // [rsp+0h] [rbp-C0h]
  __int64 v63; // [rsp+8h] [rbp-B8h]
  int v64; // [rsp+8h] [rbp-B8h]
  __int64 v65; // [rsp+10h] [rbp-B0h]
  __int64 v66; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v68; // [rsp+28h] [rbp-98h] BYREF
  char v69; // [rsp+37h] [rbp-89h] BYREF
  __int64 v70; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v71[4]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v72; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int8 **v73; // [rsp+68h] [rbp-58h]
  __int64 v74; // [rsp+70h] [rbp-50h]
  _QWORD v75[9]; // [rsp+78h] [rbp-48h] BYREF

  v5 = *(unsigned int *)(a3 + 152);
  v68 = a4;
  v6 = *(_QWORD *)(a3 + 136);
  if ( !(_DWORD)v5 )
    return 0;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v9 = (__int64 *)(v6 + 24LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    v18 = 1;
    while ( v10 != -4096 )
    {
      v49 = v18 + 1;
      v8 = (v5 - 1) & (v18 + v8);
      v9 = (__int64 *)(v6 + 24LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v18 = v49;
    }
    return 0;
  }
LABEL_3:
  if ( v9 == (__int64 *)(v6 + 24 * v5) )
    return 0;
  v11 = v9[1];
  if ( !v11 )
    return 0;
  v12 = v68;
  if ( *(unsigned int *)(v11 + 8) - ((unsigned __int64)(v68 == 0) - 1) <= 1 )
    return 0;
  if ( v68 )
  {
LABEL_7:
    v13 = *v12;
    if ( (unsigned __int8)v13 <= 0x1Cu )
      goto LABEL_8;
    if ( (unsigned __int8)(v13 - 34) > 0x33u )
      goto LABEL_8;
    v16 = 0x8000000000041LL;
    if ( !_bittest64(&v16, (unsigned int)(v13 - 34)) )
      goto LABEL_8;
    if ( v13 == 40 )
    {
      v17 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v12);
    }
    else
    {
      v17 = -32;
      if ( v13 != 85 )
      {
        v17 = -96;
        if ( v13 != 34 )
LABEL_95:
          BUG();
      }
    }
    if ( (v12[7] & 0x80u) != 0 )
    {
      v37 = sub_BD2BC0((__int64)v12);
      v39 = v37 + v38;
      if ( (v12[7] & 0x80u) == 0 )
      {
        if ( !(unsigned int)(v39 >> 4) )
          goto LABEL_55;
      }
      else
      {
        if ( !(unsigned int)((v39 - sub_BD2BC0((__int64)v12)) >> 4) )
          goto LABEL_55;
        if ( (v12[7] & 0x80u) != 0 )
        {
          v40 = *(_DWORD *)(sub_BD2BC0((__int64)v12) + 8);
          if ( (v12[7] & 0x80u) == 0 )
            BUG();
          v41 = sub_BD2BC0((__int64)v12);
          v17 -= 32LL * (unsigned int)(*(_DWORD *)(v41 + v42 - 4) - v40);
          goto LABEL_55;
        }
      }
      BUG();
    }
LABEL_55:
    if ( !(32LL * (*((_DWORD *)v12 + 1) & 0x7FFFFFF) + v17)
      || *(_QWORD *)(*(_QWORD *)(a1 + 72) + 3184LL) != *(_QWORD *)(*(_QWORD *)&v12[-32
                                                                                 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)]
                                                                 + 8LL) )
    {
      goto LABEL_8;
    }
    v43 = *(_QWORD *)(a1 + 40);
    v71[2] = &v70;
    v69 = 1;
    v70 = 0;
    v71[0] = a3;
    v71[1] = a2;
    v71[3] = &v69;
    sub_26807A0(a3, v43, (__int64 (__fastcall *)(__int64, _QWORD, __int64))sub_2673850, (__int64)v71);
    v44 = v70;
    if ( v70 && v69 )
    {
      v45 = &v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
      if ( !*(_QWORD *)v45 || (v46 = *((_QWORD *)v45 + 1), (**((_QWORD **)v45 + 2) = v46) == 0) )
      {
        *(_QWORD *)v45 = v44;
        goto LABEL_63;
      }
    }
    else
    {
      v57 = *(_QWORD *)(a1 + 72);
      if ( !*(_QWORD *)(v57 + 960) )
      {
        v60 = *(_QWORD *)(a2 + 80);
        if ( !v60 )
          BUG();
        v61 = *(unsigned __int8 ***)(v60 + 32);
        v72 = v60 - 24;
        v73 = v61;
        LOWORD(v74) = 1;
        v75[0] = 0;
        sub_2677420(v57 + 400, (__int64)&v72);
        sub_9C6650(v75);
        v57 = *(_QWORD *)(a1 + 72);
      }
      v58 = sub_3135D70(v57 + 400, &v72);
      v44 = sub_313A9F0(*(_QWORD *)(a1 + 72) + 400LL, v58, (unsigned int)v72, 0, 0);
      v45 = &v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
      if ( !*(_QWORD *)v45 || (v46 = *((_QWORD *)v45 + 1), (**((_QWORD **)v45 + 2) = v46) == 0) )
      {
LABEL_62:
        *(_QWORD *)v45 = v44;
        if ( v44 )
        {
LABEL_63:
          v47 = *(_QWORD *)(v44 + 16);
          *((_QWORD *)v45 + 1) = v47;
          if ( v47 )
            *(_QWORD *)(v47 + 16) = v45 + 8;
          *((_QWORD *)v45 + 2) = v44 + 16;
          *(_QWORD *)(v44 + 16) = v45;
        }
LABEL_8:
        v73 = &v68;
        LOBYTE(v71[0]) = 0;
        v14 = *(_QWORD *)(a1 + 40);
        v75[0] = a1;
        v72 = a3;
        v74 = a2;
        v75[1] = v71;
        sub_26807A0(a3, v14, (__int64 (__fastcall *)(__int64, _QWORD, __int64))sub_267BC80, (__int64)&v72);
        return LOBYTE(v71[0]);
      }
    }
    *(_QWORD *)(v46 + 16) = *((_QWORD *)v45 + 2);
    goto LABEL_62;
  }
  v19 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 240LL);
  v20 = *(_QWORD *)v19;
  if ( !*(_QWORD *)v19 )
    return 0;
  if ( !*(_BYTE *)(v19 + 16) )
  {
    v21 = sub_BC1CD0(v20, &unk_4F81450, a2);
    v22 = *(__int64 **)v11;
    v62 = v21 + 8;
    v65 = *(_QWORD *)v11 + 8LL * *(unsigned int *)(v11 + 8);
    if ( v65 != *(_QWORD *)v11 )
      goto LABEL_21;
LABEL_30:
    if ( v68 )
    {
      sub_B444E0(v68, (__int64)(v12 + 24), 0);
      v12 = v68;
      goto LABEL_7;
    }
    return 0;
  }
  v50 = *(unsigned int *)(v20 + 88);
  v51 = *(_QWORD *)(v20 + 72);
  if ( !(_DWORD)v50 )
    return 0;
  v52 = 1;
  v53 = 0xBF58476D1CE4E5B9LL
      * (((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32)
       | ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  for ( i = (v50 - 1) & ((v53 >> 31) ^ v53); ; i = (v50 - 1) & v56 )
  {
    v55 = v51 + 24LL * i;
    if ( *(_UNKNOWN **)v55 == &unk_4F81450 && a2 == *(_QWORD *)(v55 + 8) )
    {
      if ( v55 == v51 + 24 * v50 )
        return 0;
      v59 = *(_QWORD *)(*(_QWORD *)(v55 + 16) + 24LL);
      if ( !v59 )
        return 0;
      v22 = *(__int64 **)v11;
      v62 = v59 + 8;
      v65 = *(_QWORD *)v11 + 8LL * *(unsigned int *)(v11 + 8);
      if ( v65 == *(_QWORD *)v11 )
        return 0;
      while ( 1 )
      {
LABEL_21:
        v23 = (unsigned __int8 *)sub_266E210(*v22, a3);
        v24 = v23;
        if ( !v23 )
          goto LABEL_29;
        if ( v12 )
        {
          v25 = sub_B1A110(v62, (__int64)v12, (__int64)v23);
          v26 = *v24;
          v12 = (unsigned __int8 *)v25;
          v27 = v26 - 29;
          if ( v26 == 40 )
            goto LABEL_67;
        }
        else
        {
          v48 = *v23;
          v12 = v23;
          v27 = v48 - 29;
          if ( v48 == 40 )
          {
LABEL_67:
            v66 = 32LL * (unsigned int)sub_B491D0((__int64)v24);
            goto LABEL_33;
          }
        }
        v66 = 0;
        if ( v27 != 56 )
        {
          if ( v27 != 5 )
            goto LABEL_95;
          v66 = 64;
        }
LABEL_33:
        if ( (v24[7] & 0x80u) == 0 )
          goto LABEL_47;
        v28 = sub_BD2BC0((__int64)v24);
        v63 = v29 + v28;
        if ( (v24[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v63 >> 4) )
LABEL_91:
            BUG();
LABEL_47:
          v32 = 0;
          goto LABEL_39;
        }
        if ( !(unsigned int)((v63 - sub_BD2BC0((__int64)v24)) >> 4) )
          goto LABEL_47;
        if ( (v24[7] & 0x80u) == 0 )
          goto LABEL_91;
        v64 = *(_DWORD *)(sub_BD2BC0((__int64)v24) + 8);
        if ( (v24[7] & 0x80u) == 0 )
          BUG();
        v30 = sub_BD2BC0((__int64)v24);
        v32 = 32LL * (unsigned int)(*(_DWORD *)(v30 + v31 - 4) - v64);
LABEL_39:
        v33 = *((_DWORD *)v24 + 1) & 0x7FFFFFF;
        v34 = (32 * v33 - 32 - v66 - v32) >> 5;
        if ( !(_DWORD)v34 )
          goto LABEL_27;
        if ( *(_QWORD *)(*(_QWORD *)&v24[-32 * v33] + 8LL) == *(_QWORD *)(*(_QWORD *)(a1 + 72) + 3184LL) )
        {
          if ( (_DWORD)v34 != 1 )
          {
            v35 = &v24[-32 * v33 + 32];
            v36 = (__int64)&v24[32 * ((unsigned int)(v34 - 2) - v33) + 64];
            while ( **(_BYTE **)v35 <= 0x1Cu )
            {
              v35 += 32;
              if ( v35 == (unsigned __int8 *)v36 )
                goto LABEL_27;
            }
            goto LABEL_29;
          }
LABEL_27:
          if ( !v68 )
            v68 = v24;
        }
LABEL_29:
        if ( (__int64 *)v65 == ++v22 )
          goto LABEL_30;
      }
    }
    if ( *(_QWORD *)v55 == -4096 && *(_QWORD *)(v55 + 8) == -4096 )
      break;
    v56 = v52 + i;
    ++v52;
  }
  return 0;
}
