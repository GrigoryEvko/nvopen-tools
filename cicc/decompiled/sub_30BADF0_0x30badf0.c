// Function: sub_30BADF0
// Address: 0x30badf0
//
void __fastcall sub_30BADF0(_QWORD *a1)
{
  _QWORD *v2; // r8
  __int64 i; // r9
  __int64 v4; // rdx
  __int64 *v5; // rbx
  __int64 v6; // rdx
  __int64 *v7; // r14
  __int64 v8; // rsi
  int v9; // r13d
  __int64 v10; // rax
  _QWORD *v11; // rcx
  __int64 *v12; // rax
  _BYTE *v13; // rcx
  _QWORD *v14; // rax
  _BYTE *v15; // r10
  __int64 v16; // rax
  __int64 **v17; // rsi
  __int64 **j; // rdi
  __int64 v19; // r11
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r12
  __int64 *v23; // r13
  __int64 *v24; // r12
  signed __int64 v25; // r14
  _BYTE *v26; // r8
  int v27; // eax
  int v28; // eax
  int v29; // r13d
  __int64 *v30; // rax
  signed __int64 v31; // rcx
  __int64 *v32; // rdx
  __int64 *v33; // rcx
  __int64 v34; // rdx
  __int64 *v35; // rax
  unsigned int v36; // ecx
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 *v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // r13
  int v42; // r11d
  __int64 *v43; // rcx
  unsigned int v44; // r8d
  __int64 *v45; // rax
  __int64 v46; // rdi
  char *v47; // rax
  char *v48; // r9
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 *v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rax
  __int64 v54; // rcx
  unsigned __int64 v55; // rdx
  __int64 *v56; // rdx
  __int64 *v57; // rax
  __int64 *v58; // rax
  _QWORD *v59; // rax
  __int64 *v60; // rax
  int v61; // edi
  __int64 v62; // rax
  __int64 v63; // r8
  int v64; // esi
  __int64 *v65; // rdx
  __int64 *v66; // r10
  __int64 v67; // rdx
  int v68; // eax
  __int64 v69; // rsi
  _QWORD *v70; // [rsp+8h] [rbp-288h]
  __int64 v71; // [rsp+10h] [rbp-280h] BYREF
  __int64 v72; // [rsp+18h] [rbp-278h]
  __int64 v73; // [rsp+20h] [rbp-270h]
  unsigned int v74; // [rsp+28h] [rbp-268h]
  _BYTE *v75; // [rsp+30h] [rbp-260h] BYREF
  __int64 v76; // [rsp+38h] [rbp-258h]
  _BYTE v77[256]; // [rsp+40h] [rbp-250h] BYREF
  __int64 v78; // [rsp+140h] [rbp-150h] BYREF
  __int64 *v79; // [rsp+148h] [rbp-148h]
  __int64 v80; // [rsp+150h] [rbp-140h]
  int v81; // [rsp+158h] [rbp-138h]
  char v82; // [rsp+15Ch] [rbp-134h]
  _BYTE v83[304]; // [rsp+160h] [rbp-130h] BYREF

  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *))(*a1 + 96LL))(a1) )
    return;
  v4 = a1[1];
  v78 = 0;
  v79 = (__int64 *)v83;
  v5 = *(__int64 **)(v4 + 96);
  v6 = *(unsigned int *)(v4 + 104);
  v80 = 32;
  v81 = 0;
  v7 = &v5[v6];
  v82 = 1;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  if ( v5 == v7 )
  {
    v23 = (__int64 *)v83;
    goto LABEL_93;
  }
  do
  {
    v8 = *v5;
    v9 = *(_DWORD *)(*v5 + 48);
    if ( v9 != 1 )
      goto LABEL_5;
    v10 = *(_QWORD *)(v8 + 40);
    v11 = *(_QWORD **)v10;
    if ( *(_DWORD *)(*(_QWORD *)v10 + 8LL) != 1 )
      goto LABEL_5;
    if ( !v82 )
      goto LABEL_91;
    v12 = v79;
    v6 = (__int64)&v79[HIDWORD(v80)];
    if ( v79 != (__int64 *)v6 )
    {
      while ( v8 != *v12 )
      {
        if ( (__int64 *)v6 == ++v12 )
          goto LABEL_90;
      }
      goto LABEL_13;
    }
LABEL_90:
    if ( HIDWORD(v80) < (unsigned int)v80 )
    {
      ++HIDWORD(v80);
      *(_QWORD *)v6 = v8;
      ++v78;
    }
    else
    {
LABEL_91:
      v70 = v11;
      sub_C8CC70((__int64)&v78, v8, v6, (__int64)v11, (__int64)v2, i);
      v11 = v70;
    }
LABEL_13:
    v13 = (_BYTE *)*v11;
    LODWORD(v76) = 0;
    v75 = v13;
    if ( v74 )
    {
      v6 = (v74 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v14 = (_QWORD *)(v72 + 16 * v6);
      v15 = (_BYTE *)*v14;
      if ( v13 == (_BYTE *)*v14 )
        goto LABEL_5;
      v2 = 0;
      while ( v15 != (_BYTE *)-4096LL )
      {
        if ( v15 == (_BYTE *)-8192LL && !v2 )
          v2 = v14;
        i = (unsigned int)(v9 + 1);
        v6 = (v74 - 1) & (v9 + (_DWORD)v6);
        v14 = (_QWORD *)(v72 + 16LL * (unsigned int)v6);
        v15 = (_BYTE *)*v14;
        if ( v13 == (_BYTE *)*v14 )
          goto LABEL_5;
        ++v9;
      }
      if ( !v2 )
        v2 = v14;
    }
    else
    {
      v2 = 0;
    }
    v59 = sub_30BAC50((__int64)&v71, &v75, v2);
    *v59 = v75;
    v6 = (unsigned int)v76;
    *((_DWORD *)v59 + 2) = v76;
LABEL_5:
    ++v5;
  }
  while ( v7 != v5 );
  v16 = a1[1];
  v2 = *(_QWORD **)(v16 + 96);
  for ( i = (__int64)&v2[*(unsigned int *)(v16 + 104)]; (_QWORD *)i != v2; ++v2 )
  {
    v17 = *(__int64 ***)(*v2 + 40LL);
    for ( j = &v17[*(unsigned int *)(*v2 + 48LL)]; j != v17; ++v17 )
    {
      v19 = **v17;
      if ( v74 )
      {
        v20 = (v74 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v21 = (__int64 *)(v72 + 16LL * v20);
        v22 = *v21;
        if ( v19 == *v21 )
        {
LABEL_24:
          if ( v21 != (__int64 *)(v72 + 16LL * v74) )
            ++*((_DWORD *)v21 + 2);
        }
        else
        {
          v28 = 1;
          while ( v22 != -4096 )
          {
            v29 = v28 + 1;
            v20 = (v74 - 1) & (v28 + v20);
            v21 = (__int64 *)(v72 + 16LL * v20);
            v22 = *v21;
            if ( v19 == *v21 )
              goto LABEL_24;
            v28 = v29;
          }
        }
      }
    }
  }
  v23 = v79;
  if ( !v82 )
  {
    v24 = &v79[(unsigned int)v80];
    goto LABEL_30;
  }
LABEL_93:
  v24 = &v23[HIDWORD(v80)];
LABEL_30:
  if ( v23 == v24 )
  {
LABEL_33:
    HIDWORD(v76) = 32;
    v75 = v77;
LABEL_34:
    LODWORD(v25) = 0;
    v26 = v77;
    v27 = 0;
    goto LABEL_52;
  }
  while ( (unsigned __int64)*v23 >= 0xFFFFFFFFFFFFFFFELL )
  {
    if ( ++v23 == v24 )
      goto LABEL_33;
  }
  v75 = v77;
  v76 = 0x2000000000LL;
  if ( v23 == v24 )
    goto LABEL_34;
  v30 = v23;
  v31 = 0;
  while ( 1 )
  {
    v32 = v30 + 1;
    if ( v30 + 1 == v24 )
      break;
    while ( 1 )
    {
      v30 = v32;
      if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v24 == ++v32 )
        goto LABEL_44;
    }
    ++v31;
    if ( v32 == v24 )
      goto LABEL_45;
  }
LABEL_44:
  ++v31;
LABEL_45:
  v25 = v31;
  v33 = (__int64 *)v77;
  if ( v25 > 32 )
  {
    sub_C8D5F0((__int64)&v75, v77, v25, 8u, (__int64)v2, i);
    v33 = (__int64 *)&v75[8 * (unsigned int)v76];
  }
  v34 = *v23;
  do
  {
    v35 = v23 + 1;
    *v33++ = v34;
    if ( v23 + 1 == v24 )
      break;
    while ( 1 )
    {
      v34 = *v35;
      v23 = v35;
      if ( (unsigned __int64)*v35 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v24 == ++v35 )
        goto LABEL_51;
    }
  }
  while ( v35 != v24 );
LABEL_51:
  v27 = v76;
  v26 = v75;
LABEL_52:
  LODWORD(v76) = v25 + v27;
  v36 = v25 + v27;
LABEL_53:
  while ( 2 )
  {
    if ( v36 )
    {
LABEL_54:
      v37 = v36--;
      v38 = *(_QWORD *)&v26[8 * v37 - 8];
      LODWORD(v76) = v36;
      if ( !v82 )
      {
        v58 = sub_C8CA60((__int64)&v78, v38);
        if ( v58 )
        {
          *v58 = -2;
          ++v81;
          ++v78;
          goto LABEL_60;
        }
        goto LABEL_78;
      }
      v39 = &v79[HIDWORD(v80)];
      if ( v79 == v39 )
        continue;
      v40 = v79;
      while ( v38 != *v40 )
      {
        if ( v39 == ++v40 )
          goto LABEL_53;
      }
      --HIDWORD(v80);
      *v40 = v79[HIDWORD(v80)];
      ++v78;
LABEL_60:
      v41 = **(_QWORD **)(*(_QWORD *)(v38 + 40) + 8LL * *(unsigned int *)(v38 + 48) - 8);
      if ( v74 )
      {
        v42 = 1;
        v43 = 0;
        v44 = (v74 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v45 = (__int64 *)(v72 + 16LL * v44);
        v46 = *v45;
        if ( v41 == *v45 )
        {
LABEL_62:
          if ( *((_DWORD *)v45 + 2) == 1 )
          {
            if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, __int64, __int64, __int64 *))(*a1 + 104LL))(
                   a1,
                   v38,
                   v41,
                   v43) )
            {
              v47 = sub_30B9560(
                      *(char **)(v41 + 40),
                      (char *)(*(_QWORD *)(v41 + 40) + 8LL * *(unsigned int *)(v41 + 48)),
                      v38);
              if ( v48 == v47 )
              {
                (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 112LL))(a1, v38, v41);
                if ( v82 )
                {
                  v51 = &v79[HIDWORD(v80)];
                  if ( v79 != v51 )
                  {
                    v52 = v79;
                    while ( v41 != *v52 )
                    {
                      if ( v51 == ++v52 )
                        goto LABEL_78;
                    }
                    --HIDWORD(v80);
                    *v52 = v79[HIDWORD(v80)];
                    ++v78;
LABEL_71:
                    v53 = (unsigned int)v76;
                    v54 = HIDWORD(v76);
                    v55 = (unsigned int)v76 + 1LL;
                    if ( v55 > HIDWORD(v76) )
                    {
                      sub_C8D5F0((__int64)&v75, v77, v55, 8u, v49, v50);
                      v53 = (unsigned int)v76;
                    }
                    v56 = (__int64 *)v75;
                    *(_QWORD *)&v75[8 * v53] = v38;
                    LODWORD(v76) = v76 + 1;
                    if ( v82 )
                    {
                      v57 = v79;
                      v54 = HIDWORD(v80);
                      v56 = &v79[HIDWORD(v80)];
                      if ( v79 == v56 )
                      {
LABEL_126:
                        if ( HIDWORD(v80) >= (unsigned int)v80 )
                          goto LABEL_127;
                        ++HIDWORD(v80);
                        *v56 = v38;
                        ++v78;
                      }
                      else
                      {
                        while ( v38 != *v57 )
                        {
                          if ( v56 == ++v57 )
                            goto LABEL_126;
                        }
                      }
                    }
                    else
                    {
LABEL_127:
                      sub_C8CC70((__int64)&v78, v38, (__int64)v56, v54, v49, v50);
                    }
                  }
                }
                else
                {
                  v60 = sub_C8CA60((__int64)&v78, v41);
                  if ( v60 )
                  {
                    *v60 = -2;
                    ++v81;
                    ++v78;
                    goto LABEL_71;
                  }
                }
              }
            }
          }
LABEL_78:
          v36 = v76;
          v26 = v75;
LABEL_79:
          if ( !v36 )
            break;
          goto LABEL_54;
        }
        while ( v46 != -4096 )
        {
          if ( v46 == -8192 && !v43 )
            v43 = v45;
          v44 = (v74 - 1) & (v42 + v44);
          v45 = (__int64 *)(v72 + 16LL * v44);
          v46 = *v45;
          if ( v41 == *v45 )
            goto LABEL_62;
          ++v42;
        }
        if ( !v43 )
          v43 = v45;
        ++v71;
        v61 = v73 + 1;
        if ( 4 * ((int)v73 + 1) < 3 * v74 )
        {
          if ( v74 - HIDWORD(v73) - v61 <= v74 >> 3 )
          {
            sub_30BAA70((__int64)&v71, v74);
            if ( !v74 )
            {
LABEL_144:
              LODWORD(v73) = v73 + 1;
              BUG();
            }
            v66 = 0;
            LODWORD(v67) = (v74 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
            v61 = v73 + 1;
            v68 = 1;
            v43 = (__int64 *)(v72 + 16LL * (unsigned int)v67);
            v69 = *v43;
            if ( v41 != *v43 )
            {
              while ( v69 != -4096 )
              {
                if ( !v66 && v69 == -8192 )
                  v66 = v43;
                v67 = (v74 - 1) & ((_DWORD)v67 + v68);
                v43 = (__int64 *)(v72 + 16 * v67);
                v69 = *v43;
                if ( v41 == *v43 )
                  goto LABEL_109;
                ++v68;
              }
              if ( v66 )
                v43 = v66;
            }
          }
          goto LABEL_109;
        }
      }
      else
      {
        ++v71;
      }
      sub_30BAA70((__int64)&v71, 2 * v74);
      if ( !v74 )
        goto LABEL_144;
      v62 = (v74 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v61 = v73 + 1;
      v43 = (__int64 *)(v72 + 16 * v62);
      v63 = *v43;
      if ( v41 != *v43 )
      {
        v64 = 1;
        v65 = 0;
        while ( v63 != -4096 )
        {
          if ( !v65 && v63 == -8192 )
            v65 = v43;
          LODWORD(v62) = (v74 - 1) & (v64 + v62);
          v43 = (__int64 *)(v72 + 16LL * (unsigned int)v62);
          v63 = *v43;
          if ( v41 == *v43 )
            goto LABEL_109;
          ++v64;
        }
        if ( v65 )
          v43 = v65;
      }
LABEL_109:
      LODWORD(v73) = v61;
      if ( *v43 != -4096 )
        --HIDWORD(v73);
      *v43 = v41;
      *((_DWORD *)v43 + 2) = 0;
      v36 = v76;
      v26 = v75;
      goto LABEL_79;
    }
    break;
  }
  if ( v26 != v77 )
    _libc_free((unsigned __int64)v26);
  sub_C7D6A0(v72, 16LL * v74, 8);
  if ( !v82 )
    _libc_free((unsigned __int64)v79);
}
