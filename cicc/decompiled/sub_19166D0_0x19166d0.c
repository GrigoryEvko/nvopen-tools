// Function: sub_19166D0
// Address: 0x19166d0
//
__int64 __fastcall sub_19166D0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, char a5)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 *v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rbx
  unsigned __int8 v16; // dl
  __int64 v17; // rax
  unsigned int v18; // eax
  unsigned __int8 v19; // dl
  unsigned int v20; // r15d
  unsigned __int8 v21; // al
  unsigned int v22; // eax
  __int64 v23; // rcx
  unsigned __int8 v24; // dl
  __int64 v25; // rax
  int v26; // ecx
  __int64 v27; // rsi
  __int64 v28; // r8
  int v29; // ecx
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  int v35; // eax
  int v36; // r9d
  unsigned int v37; // r15d
  bool v38; // zf
  unsigned __int8 v39; // al
  bool v40; // r15
  __int64 v42; // r8
  __int64 v43; // r15
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rax
  __int64 v48; // r11
  __int64 v49; // r8
  int v50; // edi
  __int64 v51; // rsi
  unsigned int v52; // r15d
  __int64 v53; // r14
  unsigned int v54; // eax
  int v55; // ebx
  __int64 v56; // rax
  __int64 *v57; // rax
  int v58; // edi
  __int64 v59; // rax
  _DWORD *v60; // rax
  __int64 v61; // rdx
  int v62; // eax
  int v63; // eax
  int v64; // r9d
  void *v65; // rax
  int v66; // r9d
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 *v69; // rax
  __int64 v70; // [rsp+8h] [rbp-B8h]
  __int64 v71; // [rsp+10h] [rbp-B0h]
  __int64 v72; // [rsp+10h] [rbp-B0h]
  __int64 v73; // [rsp+10h] [rbp-B0h]
  unsigned int v74; // [rsp+18h] [rbp-A8h]
  __int64 v75; // [rsp+18h] [rbp-A8h]
  __int64 v76; // [rsp+18h] [rbp-A8h]
  __int64 v77; // [rsp+18h] [rbp-A8h]
  __int64 v78; // [rsp+28h] [rbp-98h]
  unsigned __int8 v80; // [rsp+37h] [rbp-89h]
  _QWORD *v82; // [rsp+40h] [rbp-80h] BYREF
  __int64 v83; // [rsp+48h] [rbp-78h]
  _QWORD v84[14]; // [rsp+50h] [rbp-70h] BYREF

  v7 = a4[1];
  v82 = v84;
  v84[0] = a2;
  v84[1] = a3;
  v83 = 0x400000001LL;
  v8 = sub_157F0B0(v7);
  v80 = 0;
  v9 = v84;
  v78 = v8;
  v10 = 1;
  v11 = a1 + 152;
  do
  {
    while ( 1 )
    {
      v12 = v10--;
      v13 = &v9[2 * v12 - 2];
      v14 = *v13;
      v15 = v13[1];
      LODWORD(v83) = v10;
      if ( v14 != v15 )
        break;
LABEL_34:
      if ( !v10 )
        goto LABEL_35;
    }
    v16 = *(_BYTE *)(v14 + 16);
    if ( v16 <= 0x10u )
    {
      if ( *(_BYTE *)(v15 + 16) <= 0x10u )
        goto LABEL_34;
    }
    else if ( v16 != 17 || *(_BYTE *)(v15 + 16) <= 0x10u )
    {
      v17 = v15;
      v15 = v14;
      v14 = v17;
    }
    v18 = sub_1911FD0(v11, v15);
    v19 = *(_BYTE *)(v14 + 16);
    v20 = v18;
    v21 = *(_BYTE *)(v15 + 16);
    if ( v21 == 17 )
    {
      if ( v19 != 17 )
      {
LABEL_41:
        if ( v19 > 0x17u )
          goto LABEL_21;
LABEL_42:
        if ( v78 )
          goto LABEL_43;
        goto LABEL_21;
      }
    }
    else
    {
      if ( v21 <= 0x17u )
        goto LABEL_41;
      if ( v19 <= 0x17u )
        goto LABEL_42;
    }
    v22 = sub_1911FD0(v11, v14);
    if ( v20 >= v22 )
    {
      v24 = *(_BYTE *)(v14 + 16);
    }
    else
    {
      v23 = v15;
      v24 = *(_BYTE *)(v15 + 16);
      v20 = v22;
      v15 = v14;
      v14 = v23;
    }
    if ( v24 <= 0x17u )
      goto LABEL_42;
    if ( *(_BYTE *)(v15 + 16) > 0x17u )
    {
      v25 = *(_QWORD *)(a1 + 16);
      if ( v25 )
      {
        v26 = *(_DWORD *)(v25 + 24);
        if ( v26 )
        {
          v27 = *(_QWORD *)(v15 + 40);
          v28 = *(_QWORD *)(v25 + 8);
          v29 = v26 - 1;
          v30 = v29 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v31 = (__int64 *)(v28 + 16LL * v30);
          v32 = *v31;
          if ( v27 == *v31 )
          {
LABEL_16:
            if ( v31[1] )
            {
              if ( (unsigned int)sub_1CED350() == 0x7FFFFFFF )
              {
                v59 = v15;
                v15 = v14;
                v14 = v59;
              }
              if ( v78 && *(_BYTE *)(v14 + 16) <= 0x17u )
LABEL_43:
                sub_1910810(a1, v20, v14, a4[1]);
            }
          }
          else
          {
            v63 = 1;
            while ( v32 != -8 )
            {
              v64 = v63 + 1;
              v30 = v29 & (v63 + v30);
              v31 = (__int64 *)(v28 + 16LL * v30);
              v32 = *v31;
              if ( v27 == *v31 )
                goto LABEL_16;
              v63 = v64;
            }
          }
        }
      }
    }
LABEL_21:
    v33 = *(_QWORD *)(v15 + 8);
    if ( !v33 || *(_QWORD *)(v33 + 8) )
    {
      v34 = *(_QWORD *)(a1 + 24);
      if ( a5 )
        v35 = sub_1AEC470(v15, v14, v34, a4);
      else
        v35 = sub_1AEC550(v15, v14, v34, *a4);
      v80 |= v35 != 0;
    }
    if ( !sub_1642F90(*(_QWORD *)v14, 1) || *(_BYTE *)(v14 + 16) != 13 )
      goto LABEL_33;
    v37 = *(_DWORD *)(v14 + 32);
    if ( v37 <= 0x40 )
    {
      v39 = *(_BYTE *)(v15 + 16);
      v40 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37) == *(_QWORD *)(v14 + 24);
      if ( !v40 )
        goto LABEL_30;
LABEL_49:
      if ( v39 == 50 )
        goto LABEL_55;
      if ( v39 == 5 )
      {
        if ( *(_WORD *)(v15 + 18) != 26 )
          goto LABEL_33;
LABEL_52:
        v42 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
        if ( !v42 )
          goto LABEL_33;
        v43 = *(_QWORD *)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
        if ( !v43 )
          goto LABEL_33;
        goto LABEL_57;
      }
LABEL_63:
      if ( v39 <= 0x17u || (unsigned __int8)(v39 - 75) > 1u )
      {
LABEL_33:
        v10 = v83;
        v9 = v82;
        goto LABEL_34;
      }
      v48 = *(_QWORD *)(v15 - 48);
      v49 = *(_QWORD *)(v15 - 24);
      v50 = *(_WORD *)(v15 + 18) & 0x7FFF;
      if ( v40 )
      {
        if ( v50 != 32 )
          goto LABEL_74;
      }
      else if ( v50 != 33 )
      {
        goto LABEL_67;
      }
      v56 = (unsigned int)v83;
      if ( (unsigned int)v83 >= HIDWORD(v83) )
      {
        v73 = *(_QWORD *)(v15 - 48);
        v77 = *(_QWORD *)(v15 - 24);
        sub_16CD150((__int64)&v82, v84, 0, 16, v49, v36);
        v56 = (unsigned int)v83;
        v48 = v73;
        v49 = v77;
      }
      v57 = &v82[2 * v56];
      *v57 = v48;
      v57[1] = v49;
      v58 = *(unsigned __int16 *)(v15 + 18);
      LODWORD(v83) = v83 + 1;
      v50 = v58 & 0xFFFF7FFF;
      if ( !v40 )
      {
LABEL_67:
        if ( v50 != 14 )
          goto LABEL_70;
        goto LABEL_68;
      }
LABEL_74:
      if ( v50 != 1 )
        goto LABEL_70;
LABEL_68:
      if ( *(_BYTE *)(v49 + 16) == 14 )
      {
        v72 = v48;
        v75 = v49;
        v65 = sub_16982C0();
        v49 = v75;
        v48 = v72;
        v67 = *(void **)(v75 + 32) == v65 ? *(_QWORD *)(v75 + 40) + 8LL : v75 + 32;
        if ( (*(_BYTE *)(v67 + 18) & 7) != 3 )
        {
          v68 = (unsigned int)v83;
          if ( (unsigned int)v83 >= HIDWORD(v83) )
          {
            sub_16CD150((__int64)&v82, v84, 0, 16, v75, v66);
            v68 = (unsigned int)v83;
            v48 = v72;
            v49 = v75;
          }
          v69 = &v82[2 * v68];
          *v69 = v48;
          v69[1] = v49;
          LODWORD(v83) = v83 + 1;
        }
      }
      v50 = *(_WORD *)(v15 + 18) & 0x7FFF;
LABEL_70:
      v70 = v48;
      v71 = v49;
      v51 = !v40;
      v52 = sub_15FF0F0(v50);
      v53 = sub_15A0680(*(_QWORD *)v15, v51, 0);
      v74 = *(_DWORD *)(a1 + 360);
      v54 = sub_1916670(v11, (unsigned int)*(unsigned __int8 *)(v15 + 16) - 24, v52, v70, v71);
      v55 = v54;
      if ( v54 < v74 )
      {
        v60 = sub_1910330(a1, a4[1], v54);
        if ( v60 )
        {
          if ( *((_BYTE *)v60 + 16) > 0x17u )
          {
            v61 = *(_QWORD *)(a1 + 24);
            if ( a5 )
              v62 = sub_1AEC470(v60, v53, v61, a4);
            else
              v62 = sub_1AEC550(v60, v53, v61, *a4);
            v80 |= v62 != 0;
          }
        }
      }
      if ( v78 )
      {
        sub_1910810(a1, v55, v53, a4[1]);
        v10 = v83;
        v9 = v82;
        goto LABEL_34;
      }
      goto LABEL_33;
    }
    v38 = v37 == (unsigned int)sub_16A58F0(v14 + 24);
    v39 = *(_BYTE *)(v15 + 16);
    v40 = v38;
    if ( v38 )
      goto LABEL_49;
LABEL_30:
    if ( v39 != 51 )
    {
      if ( v39 != 5 )
        goto LABEL_63;
      if ( *(_WORD *)(v15 + 18) != 27 )
        goto LABEL_33;
      goto LABEL_52;
    }
LABEL_55:
    v42 = *(_QWORD *)(v15 - 48);
    if ( !v42 )
      goto LABEL_33;
    v43 = *(_QWORD *)(v15 - 24);
    if ( !v43 )
      goto LABEL_33;
LABEL_57:
    v44 = (unsigned int)v83;
    if ( (unsigned int)v83 >= HIDWORD(v83) )
    {
      v76 = v42;
      sub_16CD150((__int64)&v82, v84, 0, 16, v42, v36);
      v44 = (unsigned int)v83;
      v42 = v76;
    }
    v45 = &v82[2 * v44];
    *v45 = v42;
    v45[1] = v14;
    v46 = (unsigned int)(v83 + 1);
    LODWORD(v83) = v46;
    if ( HIDWORD(v83) <= (unsigned int)v46 )
    {
      sub_16CD150((__int64)&v82, v84, 0, 16, v42, v36);
      v46 = (unsigned int)v83;
    }
    v47 = &v82[2 * v46];
    *v47 = v43;
    v9 = v82;
    v47[1] = v14;
    v10 = v83 + 1;
    LODWORD(v83) = v10;
  }
  while ( v10 );
LABEL_35:
  if ( v9 != v84 )
    _libc_free((unsigned __int64)v9);
  return v80;
}
