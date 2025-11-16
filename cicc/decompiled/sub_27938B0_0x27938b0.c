// Function: sub_27938B0
// Address: 0x27938b0
//
__int64 __fastcall sub_27938B0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // rax
  unsigned int v5; // esi
  int v6; // r14d
  __int64 v7; // r8
  int v8; // r11d
  unsigned __int8 **v9; // rcx
  unsigned int v10; // edi
  _QWORD *v11; // rax
  unsigned __int8 *v12; // rdx
  _DWORD *v13; // rax
  unsigned int v14; // r13d
  unsigned int v16; // eax
  int v17; // r13d
  unsigned __int64 v18; // rdi
  int v19; // eax
  int v20; // edi
  __int64 v21; // r8
  unsigned int v22; // esi
  int v23; // edx
  unsigned __int8 *v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  int v27; // edx
  int v28; // r13d
  int v29; // eax
  int v30; // eax
  int v31; // esi
  __int64 v32; // rdi
  unsigned __int8 **v33; // r8
  unsigned int v34; // r13d
  int v35; // r9d
  unsigned __int8 *v36; // rax
  unsigned __int8 *v37; // r14
  const __m128i **v38; // rax
  const __m128i *v39; // r15
  __int64 *v40; // r13
  unsigned __int64 v41; // rax
  char v42; // al
  int v43; // edx
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r13
  int v48; // r13d
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  int v52; // edx
  __int64 v53; // r13
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // r15
  unsigned int v61; // eax
  int v62; // r13d
  __int64 v63; // rsi
  unsigned __int8 *v64; // rax
  unsigned __int8 *v65; // r13
  int v66; // r14d
  __int64 v67; // r15
  unsigned int v68; // eax
  int v69; // r14d
  int v70; // r10d
  unsigned __int8 **v71; // r9
  __int64 v72; // [rsp+8h] [rbp-88h]
  int v73; // [rsp+8h] [rbp-88h]
  __int64 v74; // [rsp+8h] [rbp-88h]
  __int64 v75; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v76; // [rsp+18h] [rbp-78h] BYREF
  __int64 v77[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v78; // [rsp+30h] [rbp-60h]
  _BYTE v79[80]; // [rsp+40h] [rbp-50h] BYREF

  v4 = sub_B43CB0((__int64)a2);
  if ( !(unsigned __int8)sub_B2D610(v4, 49) )
  {
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 6) || (unsigned __int8)sub_B49560((__int64)a2, 6) )
      goto LABEL_12;
    if ( !(unsigned int)sub_CF5CA0(*(_QWORD *)(a1 + 184), (__int64)a2) )
    {
      sub_2793480((__int64)v77, a1, (__int64 *)a2);
      v76 = a2;
      v14 = sub_2792D30(a1, (__int64)v77);
      *(_DWORD *)sub_2791170(a1, (__int64 *)&v76) = v14;
      v18 = (unsigned __int64)v78;
      if ( v78 == v79 )
        return v14;
LABEL_14:
      _libc_free(v18);
      return v14;
    }
    if ( !*(_QWORD *)(a1 + 192)
      || (v16 = sub_CF5CA0(*(_QWORD *)(a1 + 184), (__int64)a2),
          (((unsigned __int8)((v16 >> 4) | v16 | (v16 >> 2)) | (unsigned __int8)(v16 >> 6)) & 2) != 0) )
    {
LABEL_12:
      v17 = *(_DWORD *)(a1 + 208);
      v77[0] = (__int64)a2;
      *(_DWORD *)sub_2791170(a1, v77) = v17;
      v14 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)(a1 + 208) = v14 + 1;
      return v14;
    }
    sub_2793480((__int64)v77, a1, (__int64 *)a2);
    v25 = sub_2792D30(a1, (__int64)v77);
    if ( BYTE4(v25) )
    {
      v76 = a2;
      v14 = v25;
      *(_DWORD *)sub_2791170(a1, (__int64 *)&v76) = v25;
      goto LABEL_25;
    }
    v26 = sub_1037A30(*(_QWORD *)(a1 + 192), a2, 1);
    v27 = v26 & 7;
    if ( v27 == 2 )
    {
      v64 = (unsigned __int8 *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
      v65 = v64;
      if ( *v64 == 85 )
      {
        v66 = sub_A17190(v64);
        if ( v66 == (unsigned int)sub_A17190(a2) )
        {
          v67 = 0;
          v68 = sub_A17190(a2);
          v75 = v68;
          if ( v68 )
          {
            do
            {
              v69 = sub_2792F80(a1, *(_QWORD *)&a2[32 * (v67 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
              if ( v69 != (unsigned int)sub_2792F80(
                                          a1,
                                          *(_QWORD *)&v65[32 * (v67 - (*((_DWORD *)v65 + 1) & 0x7FFFFFF))]) )
                goto LABEL_24;
            }
            while ( v75 != ++v67 );
          }
          v63 = (__int64)v65;
LABEL_80:
          v76 = a2;
          v14 = sub_2792F80(a1, v63);
          *(_DWORD *)sub_2791170(a1, (__int64 *)&v76) = v14;
LABEL_25:
          v18 = (unsigned __int64)v78;
          if ( v78 == v79 )
            return v14;
          goto LABEL_14;
        }
      }
LABEL_24:
      v28 = *(_DWORD *)(a1 + 208);
      v76 = a2;
      *(_DWORD *)sub_2791170(a1, (__int64 *)&v76) = v28;
      v14 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)(a1 + 208) = v14 + 1;
      goto LABEL_25;
    }
    if ( v27 != 3 )
      goto LABEL_24;
    if ( v26 >> 61 != 1 )
      goto LABEL_24;
    v37 = 0;
    v38 = sub_10305C0(*(_QWORD *)(a1 + 192), (__int64)a2);
    v39 = v38[1];
    v40 = (__int64 *)*v38;
    if ( *v38 == v39 )
      goto LABEL_24;
    do
    {
      v41 = v40[1];
      if ( (v41 & 7) == 3 )
      {
        if ( v41 >> 61 != 1 )
          goto LABEL_24;
      }
      else
      {
        if ( v37 )
          goto LABEL_24;
        if ( (v41 & 7) != 2 )
          goto LABEL_24;
        v37 = (unsigned __int8 *)(v41 & 0xFFFFFFFFFFFFFFF8LL);
        if ( *(_BYTE *)(v41 & 0xFFFFFFFFFFFFFFF8LL) != 85 )
          goto LABEL_24;
        sub_B196A0(*(_QWORD *)(a1 + 200), *v40, *((_QWORD *)a2 + 5));
        if ( !v42 )
          goto LABEL_24;
      }
      v40 += 2;
    }
    while ( v39 != (const __m128i *)v40 );
    if ( !v37 )
      goto LABEL_24;
    v43 = *v37;
    if ( v43 == 40 )
    {
      v44 = 32LL * (unsigned int)sub_B491D0((__int64)v37);
    }
    else
    {
      v44 = 0;
      if ( v43 != 85 )
      {
        v44 = 64;
        if ( v43 != 34 )
LABEL_113:
          BUG();
      }
    }
    if ( (v37[7] & 0x80u) != 0 )
    {
      v45 = sub_BD2BC0((__int64)v37);
      v47 = v45 + v46;
      if ( (v37[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v47 >> 4) )
          goto LABEL_115;
      }
      else if ( (unsigned int)((v47 - sub_BD2BC0((__int64)v37)) >> 4) )
      {
        if ( (v37[7] & 0x80u) != 0 )
        {
          v48 = *(_DWORD *)(sub_BD2BC0((__int64)v37) + 8);
          if ( (v37[7] & 0x80u) == 0 )
            BUG();
          v49 = sub_BD2BC0((__int64)v37);
          v51 = 32LL * (unsigned int)(*(_DWORD *)(v49 + v50 - 4) - v48);
LABEL_64:
          v52 = *a2;
          v53 = (32LL * (*((_DWORD *)v37 + 1) & 0x7FFFFFF) - 32 - v44 - v51) >> 5;
          if ( v52 == 40 )
          {
            v54 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
          }
          else
          {
            v54 = 0;
            if ( v52 != 85 )
            {
              v54 = 64;
              if ( v52 != 34 )
                goto LABEL_113;
            }
          }
          if ( (a2[7] & 0x80u) != 0 )
          {
            v55 = sub_BD2BC0((__int64)a2);
            v72 = v56 + v55;
            if ( (a2[7] & 0x80u) == 0 )
            {
              if ( (unsigned int)(v72 >> 4) )
                goto LABEL_111;
            }
            else if ( (unsigned int)((v72 - sub_BD2BC0((__int64)a2)) >> 4) )
            {
              if ( (a2[7] & 0x80u) != 0 )
              {
                v73 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
                if ( (a2[7] & 0x80u) == 0 )
                  BUG();
                v57 = sub_BD2BC0((__int64)a2);
                v59 = 32LL * (unsigned int)(*(_DWORD *)(v57 + v58 - 4) - v73);
LABEL_75:
                if ( (_DWORD)v53 == (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v54 - v59) >> 5) )
                {
                  v60 = 0;
                  v61 = sub_A17190(a2);
                  v74 = v61;
                  if ( v61 )
                  {
                    do
                    {
                      v62 = sub_2792F80(a1, *(_QWORD *)&a2[32 * (v60 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
                      if ( v62 != (unsigned int)sub_2792F80(
                                                  a1,
                                                  *(_QWORD *)&v37[32 * (v60 - (*((_DWORD *)v37 + 1) & 0x7FFFFFF))]) )
                        goto LABEL_24;
                    }
                    while ( ++v60 != v74 );
                  }
                  v63 = (__int64)v37;
                  goto LABEL_80;
                }
                goto LABEL_24;
              }
LABEL_111:
              BUG();
            }
          }
          v59 = 0;
          goto LABEL_75;
        }
LABEL_115:
        BUG();
      }
    }
    v51 = 0;
    goto LABEL_64;
  }
  v5 = *(_DWORD *)(a1 + 24);
  v6 = *(_DWORD *)(a1 + 208);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_16;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (_QWORD *)(v7 + 16LL * v10);
  v12 = (unsigned __int8 *)*v11;
  if ( a2 != (unsigned __int8 *)*v11 )
  {
    while ( v12 != (unsigned __int8 *)-4096LL )
    {
      if ( v12 == (unsigned __int8 *)-8192LL && !v9 )
        v9 = (unsigned __int8 **)v11;
      v10 = (v5 - 1) & (v8 + v10);
      v11 = (_QWORD *)(v7 + 16LL * v10);
      v12 = (unsigned __int8 *)*v11;
      if ( a2 == (unsigned __int8 *)*v11 )
        goto LABEL_4;
      ++v8;
    }
    if ( !v9 )
      v9 = (unsigned __int8 **)v11;
    v29 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v23 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 20) - v23 > v5 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(a1 + 16) = v23;
        if ( *v9 != (unsigned __int8 *)-4096LL )
          --*(_DWORD *)(a1 + 20);
        *v9 = a2;
        v13 = v9 + 1;
        *((_DWORD *)v9 + 2) = 0;
        goto LABEL_5;
      }
      sub_D39D40(a1, v5);
      v30 = *(_DWORD *)(a1 + 24);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a1 + 8);
        v33 = 0;
        v34 = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v35 = 1;
        v23 = *(_DWORD *)(a1 + 16) + 1;
        v9 = (unsigned __int8 **)(v32 + 16LL * v34);
        v36 = *v9;
        if ( a2 != *v9 )
        {
          while ( v36 != (unsigned __int8 *)-4096LL )
          {
            if ( !v33 && v36 == (unsigned __int8 *)-8192LL )
              v33 = v9;
            v34 = v31 & (v35 + v34);
            v9 = (unsigned __int8 **)(v32 + 16LL * v34);
            v36 = *v9;
            if ( a2 == *v9 )
              goto LABEL_18;
            ++v35;
          }
          if ( v33 )
            v9 = v33;
        }
        goto LABEL_18;
      }
LABEL_110:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_16:
    sub_D39D40(a1, 2 * v5);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 16) + 1;
      v9 = (unsigned __int8 **)(v21 + 16LL * v22);
      v24 = *v9;
      if ( a2 != *v9 )
      {
        v70 = 1;
        v71 = 0;
        while ( v24 != (unsigned __int8 *)-4096LL )
        {
          if ( !v71 && v24 == (unsigned __int8 *)-8192LL )
            v71 = v9;
          v22 = v20 & (v70 + v22);
          v9 = (unsigned __int8 **)(v21 + 16LL * v22);
          v24 = *v9;
          if ( a2 == *v9 )
            goto LABEL_18;
          ++v70;
        }
        if ( v71 )
          v9 = v71;
      }
      goto LABEL_18;
    }
    goto LABEL_110;
  }
LABEL_4:
  v13 = v11 + 1;
LABEL_5:
  *v13 = v6;
  v14 = *(_DWORD *)(a1 + 208);
  *(_DWORD *)(a1 + 208) = v14 + 1;
  return v14;
}
