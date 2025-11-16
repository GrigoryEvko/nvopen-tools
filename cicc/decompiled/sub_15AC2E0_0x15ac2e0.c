// Function: sub_15AC2E0
// Address: 0x15ac2e0
//
__int64 __fastcall sub_15AC2E0(__int64 a1)
{
  __int64 v1; // rsi
  unsigned int v2; // r12d
  __int64 v3; // rcx
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r9
  unsigned int v14; // r8d
  unsigned int v15; // r15d
  unsigned int v16; // eax
  __int64 *v17; // r10
  __int64 *v18; // rdx
  __int64 v19; // rdi
  __int64 v21; // r11
  int j; // edx
  int v23; // edx
  __int64 *v24; // r11
  int v25; // eax
  __int64 *v26; // rax
  __int64 v27; // r8
  __int64 v28; // r8
  _QWORD *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 *v32; // r10
  __int64 v33; // r11
  __int64 v34; // rax
  __int64 v35; // rdx
  _QWORD *i; // rax
  _BYTE *v37; // r8
  _QWORD *v38; // rdi
  __int64 v39; // r8
  unsigned int v40; // edx
  __int64 v41; // rdi
  int v42; // esi
  __int64 *v43; // rcx
  int v44; // ecx
  unsigned int v45; // r15d
  __int64 *v46; // rdx
  __int64 v47; // rsi
  int v48; // r11d
  __int64 v49; // rdx
  __int64 *v50; // rdx
  __int64 v51; // [rsp+0h] [rbp-B0h]
  __int64 v52; // [rsp+8h] [rbp-A8h]
  __int64 *v53; // [rsp+8h] [rbp-A8h]
  __int64 *v54; // [rsp+10h] [rbp-A0h]
  __int64 *v55; // [rsp+10h] [rbp-A0h]
  _QWORD *v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+18h] [rbp-98h]
  __int64 *v59; // [rsp+20h] [rbp-90h]
  __int64 v60; // [rsp+20h] [rbp-90h]
  __int64 v61; // [rsp+20h] [rbp-90h]
  __int64 v62; // [rsp+20h] [rbp-90h]
  int v63; // [rsp+20h] [rbp-90h]
  int v64; // [rsp+28h] [rbp-88h]
  __int64 v65; // [rsp+28h] [rbp-88h]
  __int64 v66; // [rsp+28h] [rbp-88h]
  __int64 *v67; // [rsp+28h] [rbp-88h]
  _BYTE *v68; // [rsp+28h] [rbp-88h]
  __int64 v69; // [rsp+28h] [rbp-88h]
  __int64 v70; // [rsp+28h] [rbp-88h]
  __int64 v71; // [rsp+30h] [rbp-80h] BYREF
  __int64 v72; // [rsp+38h] [rbp-78h]
  __int64 v73; // [rsp+40h] [rbp-70h]
  unsigned int v74; // [rsp+48h] [rbp-68h]
  _BYTE *v75; // [rsp+50h] [rbp-60h] BYREF
  __int64 v76; // [rsp+58h] [rbp-58h]
  _BYTE v77[80]; // [rsp+60h] [rbp-50h] BYREF

  v1 = 0;
  v2 = 0;
  if ( sub_1626AA0(a1, 0) )
  {
    v1 = 0;
    v2 = 1;
    sub_1627150(a1, 0);
  }
  v4 = *(_QWORD *)(a1 + 80);
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v58 = a1 + 72;
  if ( v4 == a1 + 72 )
  {
    v19 = 0;
    goto LABEL_29;
  }
  do
  {
    if ( !v4 )
      BUG();
    if ( v4 + 16 != *(_QWORD *)(v4 + 24) )
    {
      v5 = v2;
      v6 = *(_QWORD *)(v4 + 24);
      v7 = v4 + 16;
      do
      {
        while ( 1 )
        {
          v8 = v6;
          v6 = *(_QWORD *)(v6 + 8);
          if ( *(_BYTE *)(v8 - 8) != 78 )
            break;
          v9 = *(_QWORD *)(v8 - 48);
          if ( *(_BYTE *)(v9 + 16) || (*(_BYTE *)(v9 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v9 + 36) - 35) > 3 )
            break;
          sub_15F20C0(v8 - 24, v1, v5, v3);
          v5 = 1;
          if ( v7 == v6 )
            goto LABEL_18;
        }
        v1 = *(_QWORD *)(v8 + 24);
        if ( v1 )
        {
          v75 = 0;
          if ( (_BYTE **)(v8 + 24) != &v75 )
          {
            sub_161E7C0(v8 + 24);
            v1 = (__int64)v75;
            *(_QWORD *)(v8 + 24) = v75;
            if ( v1 )
              sub_1623210(&v75, v1, v8 + 24);
          }
          v5 = 1;
        }
      }
      while ( v7 != v6 );
LABEL_18:
      v2 = v5;
    }
    v10 = sub_157EBA0(v4 - 24);
    v11 = v10;
    if ( v10 && (*(_QWORD *)(v10 + 48) || *(__int16 *)(v10 + 18) < 0) )
    {
      v1 = 18;
      v12 = sub_1625790(v10, 18);
      v13 = v12;
      if ( v12 )
      {
        v1 = v74;
        if ( v74 )
        {
          v14 = v74 - 1;
          v15 = ((unsigned int)v12 >> 4) ^ ((unsigned int)v12 >> 9);
          v16 = (v74 - 1) & v15;
          v17 = (__int64 *)(v72 + 16LL * v16);
          v3 = *v17;
          if ( v13 == *v17 )
          {
            v18 = (__int64 *)v17[1];
            if ( v18 )
              goto LABEL_25;
LABEL_43:
            v26 = (__int64 *)(v13 + 8 * (1LL - *(unsigned int *)(v13 + 8)));
            v27 = -8 * (1LL - *(unsigned int *)(v13 + 8));
            v3 = v27 >> 5;
            v28 = v27 >> 3;
            if ( v3 > 0 )
            {
              v18 = (__int64 *)(v13 + 8 * (1LL - *(unsigned int *)(v13 + 8)));
              while ( 1 )
              {
                v1 = *v18;
                if ( *(_BYTE *)*v18 == 5 )
                  break;
                v1 = v18[1];
                if ( *(_BYTE *)v1 == 5 )
                {
                  ++v18;
                  break;
                }
                v1 = v18[2];
                if ( *(_BYTE *)v1 == 5 )
                {
                  v18 += 2;
                  break;
                }
                v1 = v18[3];
                if ( *(_BYTE *)v1 == 5 )
                {
                  v18 += 3;
                  break;
                }
                v18 += 4;
                if ( v18 == &v26[4 * v3] )
                {
                  v1 = (v13 - (__int64)v18) >> 3;
                  goto LABEL_80;
                }
              }
              if ( (__int64 *)v13 != v18 )
                goto LABEL_56;
              goto LABEL_74;
            }
            v1 = v28;
            v18 = (__int64 *)(v13 + 8 * (1LL - *(unsigned int *)(v13 + 8)));
LABEL_80:
            if ( v1 != 2 )
            {
              if ( v1 != 3 )
              {
                if ( v1 != 1 )
                {
LABEL_83:
                  v18 = (__int64 *)v13;
LABEL_74:
                  v17[1] = (__int64)v18;
LABEL_25:
                  if ( v18 != (__int64 *)v13 )
                  {
                    v1 = 18;
                    sub_1625C10(v11, 18, v18);
                  }
                  goto LABEL_27;
                }
                goto LABEL_95;
              }
              v1 = *v18;
              if ( *(_BYTE *)*v18 == 5 )
                goto LABEL_96;
              ++v18;
            }
            v1 = *v18;
            if ( *(_BYTE *)*v18 == 5 )
              goto LABEL_96;
            ++v18;
LABEL_95:
            v1 = *v18;
            if ( *(_BYTE *)*v18 != 5 )
              goto LABEL_83;
LABEL_96:
            if ( (__int64 *)v13 == v18 )
              goto LABEL_74;
            if ( v3 > 0 )
            {
LABEL_56:
              while ( *(_BYTE *)*v26 == 5 )
              {
                if ( *(_BYTE *)v26[1] != 5 )
                {
                  ++v26;
                  goto LABEL_57;
                }
                if ( *(_BYTE *)v26[2] != 5 )
                {
                  v26 += 2;
                  goto LABEL_57;
                }
                if ( *(_BYTE *)v26[3] != 5 )
                {
                  v26 += 3;
                  goto LABEL_57;
                }
                v26 += 4;
                if ( !--v3 )
                {
                  v28 = (v13 - (__int64)v26) >> 3;
                  goto LABEL_85;
                }
              }
              goto LABEL_57;
            }
LABEL_85:
            if ( v28 != 2 )
            {
              if ( v28 != 3 )
              {
                if ( v28 != 1 )
                {
LABEL_88:
                  v18 = 0;
                  goto LABEL_74;
                }
                goto LABEL_104;
              }
              if ( *(_BYTE *)*v26 != 5 )
              {
LABEL_57:
                if ( (__int64 *)v13 != v26 )
                {
                  v75 = v77;
                  v76 = 0x400000000LL;
                  v29 = (_QWORD *)(*(_QWORD *)(v13 + 16) & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (*(_QWORD *)(v13 + 16) & 4) != 0 )
                    v29 = (_QWORD *)*v29;
                  v59 = v17;
                  v65 = v13;
                  v30 = sub_1627350(v29, 0, 0, 2, 1);
                  v31 = v65;
                  v32 = v59;
                  v33 = v30;
                  v34 = (unsigned int)v76;
                  if ( (unsigned int)v76 >= HIDWORD(v76) )
                  {
                    v57 = v33;
                    sub_16CD150(&v75, v77, 0, 8);
                    v34 = (unsigned int)v76;
                    v33 = v57;
                    v32 = v59;
                    v31 = v65;
                  }
                  *(_QWORD *)&v75[8 * v34] = v33;
                  v35 = (unsigned int)(v76 + 1);
                  LODWORD(v76) = v76 + 1;
                  for ( i = (_QWORD *)(v31 + 8 * (1LL - *(unsigned int *)(v31 + 8))); (_QWORD *)v31 != i; ++i )
                  {
                    v37 = (_BYTE *)*i;
                    if ( *(_BYTE *)*i != 5 )
                    {
                      if ( HIDWORD(v76) <= (unsigned int)v35 )
                      {
                        v51 = v33;
                        v53 = v32;
                        v56 = i;
                        v62 = v31;
                        v68 = (_BYTE *)*i;
                        sub_16CD150(&v75, v77, 0, 8);
                        v35 = (unsigned int)v76;
                        v33 = v51;
                        v32 = v53;
                        i = v56;
                        v31 = v62;
                        v37 = v68;
                      }
                      *(_QWORD *)&v75[8 * v35] = v37;
                      v35 = (unsigned int)(v76 + 1);
                      LODWORD(v76) = v76 + 1;
                    }
                  }
                  v38 = (_QWORD *)(*(_QWORD *)(v31 + 16) & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (*(_QWORD *)(v31 + 16) & 4) != 0 )
                    v38 = (_QWORD *)*v38;
                  v52 = v33;
                  v54 = v32;
                  v60 = v31;
                  v1 = 0;
                  v66 = sub_1627350(v38, v75, v35, 0, 1);
                  sub_1630830(v66, 0, v66);
                  v18 = (__int64 *)v66;
                  v13 = v60;
                  v17 = v54;
                  if ( v52 )
                  {
                    sub_16307F0(v52, 0, v66, v3, v39);
                    v17 = v54;
                    v13 = v60;
                    v18 = (__int64 *)v66;
                  }
                  if ( v75 != v77 )
                  {
                    v55 = v17;
                    v61 = v13;
                    v67 = v18;
                    _libc_free((unsigned __int64)v75);
                    v18 = v67;
                    v13 = v61;
                    v17 = v55;
                  }
                  goto LABEL_74;
                }
                goto LABEL_88;
              }
              ++v26;
            }
            if ( *(_BYTE *)*v26 == 5 )
            {
              ++v26;
LABEL_104:
              v3 = *v26;
              v18 = 0;
              if ( *(_BYTE *)*v26 == 5 )
                goto LABEL_74;
              goto LABEL_57;
            }
            goto LABEL_57;
          }
          v64 = (v74 - 1) & v15;
          v21 = *v17;
          for ( j = 1; ; j = v63 )
          {
            if ( v21 == -8 )
              goto LABEL_34;
            v48 = j + 1;
            v49 = v14 & (v64 + j);
            v63 = v48;
            v64 = v49;
            v50 = (__int64 *)(v72 + 16 * v49);
            v21 = *v50;
            if ( v13 == *v50 )
              break;
          }
          v18 = (__int64 *)v50[1];
          if ( v18 )
            goto LABEL_25;
LABEL_34:
          v23 = 1;
          v24 = 0;
          while ( v3 != -8 )
          {
            if ( v3 == -16 && !v24 )
              v24 = v17;
            v16 = v14 & (v23 + v16);
            v17 = (__int64 *)(v72 + 16LL * v16);
            v3 = *v17;
            if ( v13 == *v17 )
              goto LABEL_43;
            ++v23;
          }
          if ( v24 )
            v17 = v24;
          ++v71;
          v25 = v73 + 1;
          if ( 4 * ((int)v73 + 1) < 3 * v74 )
          {
            if ( v74 - HIDWORD(v73) - v25 <= v74 >> 3 )
            {
              v70 = v13;
              sub_15AC120((__int64)&v71, v74);
              if ( !v74 )
              {
LABEL_140:
                LODWORD(v73) = v73 + 1;
                BUG();
              }
              v44 = 1;
              v45 = (v74 - 1) & v15;
              v13 = v70;
              v46 = 0;
              v25 = v73 + 1;
              v17 = (__int64 *)(v72 + 16LL * v45);
              v47 = *v17;
              if ( v70 != *v17 )
              {
                while ( v47 != -8 )
                {
                  if ( v47 == -16 && !v46 )
                    v46 = v17;
                  v45 = (v74 - 1) & (v44 + v45);
                  v17 = (__int64 *)(v72 + 16LL * v45);
                  v47 = *v17;
                  if ( v70 == *v17 )
                    goto LABEL_40;
                  ++v44;
                }
                if ( v46 )
                  v17 = v46;
              }
            }
            goto LABEL_40;
          }
        }
        else
        {
          ++v71;
        }
        v69 = v13;
        sub_15AC120((__int64)&v71, 2 * v74);
        if ( !v74 )
          goto LABEL_140;
        v13 = v69;
        v40 = (v74 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        v25 = v73 + 1;
        v17 = (__int64 *)(v72 + 16LL * v40);
        v41 = *v17;
        if ( v69 != *v17 )
        {
          v42 = 1;
          v43 = 0;
          while ( v41 != -8 )
          {
            if ( v41 == -16 && !v43 )
              v43 = v17;
            v40 = (v74 - 1) & (v42 + v40);
            v17 = (__int64 *)(v72 + 16LL * v40);
            v41 = *v17;
            if ( v69 == *v17 )
              goto LABEL_40;
            ++v42;
          }
          if ( v43 )
            v17 = v43;
        }
LABEL_40:
        LODWORD(v73) = v25;
        if ( *v17 != -8 )
          --HIDWORD(v73);
        *v17 = v13;
        v17[1] = 0;
        goto LABEL_43;
      }
    }
LABEL_27:
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v58 != v4 );
  v19 = v72;
LABEL_29:
  j___libc_free_0(v19);
  return v2;
}
