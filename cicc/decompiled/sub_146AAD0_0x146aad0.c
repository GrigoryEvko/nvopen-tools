// Function: sub_146AAD0
// Address: 0x146aad0
//
__int64 __fastcall sub_146AAD0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v4; // r15
  __int64 *v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r13
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r12
  int v16; // esi
  __int64 v17; // rcx
  int v18; // edi
  unsigned int v20; // eax
  __int64 v21; // rdx
  int v22; // ebx
  unsigned __int8 v23; // al
  __int64 v24; // rdx
  __int64 *v25; // r9
  unsigned int v26; // esi
  __int64 *v27; // rdx
  __int64 *v28; // rdi
  int v29; // edx
  int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // r12
  int v33; // r11d
  __int64 *v34; // r10
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 *v37; // r12
  __int64 v38; // rax
  __int64 *v39; // rbx
  __int64 v40; // rdx
  __int64 *v41; // r13
  __int64 *v42; // r15
  unsigned int v43; // eax
  __int64 *v44; // r12
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // rax
  char v51; // di
  unsigned int v52; // esi
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rbx
  __int64 v57; // rdi
  unsigned __int8 v58; // al
  int v59; // r11d
  __int64 *v60; // r8
  int i; // edi
  __int64 *v62; // r10
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // [rsp+0h] [rbp-120h]
  __int64 v66; // [rsp+8h] [rbp-118h]
  __int64 v67; // [rsp+18h] [rbp-108h]
  __int64 v68; // [rsp+20h] [rbp-100h]
  int v70; // [rsp+30h] [rbp-F0h]
  __int64 v72; // [rsp+40h] [rbp-E0h]
  unsigned int v73; // [rsp+40h] [rbp-E0h]
  __int64 v75; // [rsp+50h] [rbp-D0h] BYREF
  __int64 *v76; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v77; // [rsp+60h] [rbp-C0h] BYREF
  __int64 *v78; // [rsp+68h] [rbp-B8h]
  __int64 v79; // [rsp+70h] [rbp-B0h]
  unsigned int v80; // [rsp+78h] [rbp-A8h]
  __int64 v81; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v82; // [rsp+88h] [rbp-98h]
  __int64 v83; // [rsp+90h] [rbp-90h]
  unsigned int v84; // [rsp+98h] [rbp-88h]
  __int64 *v85; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-78h]
  __int64 v87; // [rsp+B0h] [rbp-70h] BYREF
  int v88; // [rsp+B8h] [rbp-68h]

  if ( *(_BYTE *)(a3 + 16) <= 0x17u )
    return sub_1456E90(a1);
  v4 = a3;
  if ( !sub_1377F70(a2 + 56, *(_QWORD *)(a3 + 40)) )
    return sub_1456E90(a1);
  if ( *(_BYTE *)(v4 + 16) == 77 )
  {
    if ( **(_QWORD **)(a2 + 32) != *(_QWORD *)(v4 + 40) )
      return sub_1456E90(a1);
    goto LABEL_5;
  }
  if ( !(unsigned __int8)sub_1452C00(v4) )
    return sub_1456E90(a1);
  if ( *(_BYTE *)(v4 + 16) == 77 )
  {
LABEL_5:
    v72 = v4;
    goto LABEL_6;
  }
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v72 = sub_146A0D0(v4, a2, (__int64)&v85, 0);
  j___libc_free_0(v86);
  if ( !v72 )
    return sub_1456E90(a1);
LABEL_6:
  if ( (*(_DWORD *)(v72 + 20) & 0xFFFFFFF) != 2 )
    return sub_1456E90(a1);
  v77 = 0;
  v78 = 0;
  v5 = *(__int64 **)(a2 + 32);
  v79 = 0;
  v80 = 0;
  v6 = *v5;
  v68 = *v5;
  v7 = sub_13FCB50(a2);
  v8 = sub_157F280(v6);
  v10 = v9;
  if ( v8 != v9 )
  {
    while ( 1 )
    {
      v15 = sub_1454270(v8, v7);
      if ( !v15 )
        goto LABEL_11;
      v16 = v80;
      v81 = v8;
      v17 = v8;
      if ( !v80 )
        break;
      v11 = (v80 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v12 = &v78[2 * v11];
      v13 = *v12;
      if ( v8 != *v12 )
      {
        v33 = 1;
        v34 = 0;
        while ( v13 != -8 )
        {
          if ( !v34 && v13 == -16 )
            v34 = v12;
          v11 = (v80 - 1) & (v33 + v11);
          v12 = &v78[2 * v11];
          v13 = *v12;
          if ( v8 == *v12 )
            goto LABEL_10;
          ++v33;
        }
        if ( v34 )
          v12 = v34;
        ++v77;
        v18 = v79 + 1;
        if ( 4 * ((int)v79 + 1) < 3 * v80 )
        {
          if ( v80 - HIDWORD(v79) - v18 <= v80 >> 3 )
          {
LABEL_20:
            sub_146A3C0((__int64)&v77, v16);
            sub_1463C30((__int64)&v77, &v81, &v85);
            v12 = v85;
            v17 = v81;
            v18 = v79 + 1;
          }
          LODWORD(v79) = v18;
          if ( *v12 != -8 )
            --HIDWORD(v79);
          *v12 = v17;
          v12[1] = 0;
          goto LABEL_10;
        }
LABEL_19:
        v16 = 2 * v80;
        goto LABEL_20;
      }
LABEL_10:
      v12[1] = v15;
LABEL_11:
      if ( !v8 )
        BUG();
      v14 = *(_QWORD *)(v8 + 32);
      if ( !v14 )
        BUG();
      v8 = 0;
      if ( *(_BYTE *)(v14 - 8) == 77 )
        v8 = v14 - 24;
      if ( v10 == v8 )
        goto LABEL_28;
    }
    ++v77;
    goto LABEL_19;
  }
LABEL_28:
  if ( !v80 )
    goto LABEL_40;
  v20 = (v80 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
  v21 = v78[2 * v20];
  if ( v72 != v21 )
  {
    for ( i = 1; ; ++i )
    {
      if ( v21 == -8 )
        goto LABEL_40;
      v20 = (v80 - 1) & (i + v20);
      v21 = v78[2 * v20];
      if ( v72 == v21 )
        break;
    }
  }
  v22 = dword_4F9B5E0;
  v70 = dword_4F9B5E0;
  v67 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL));
  if ( !v22 )
    goto LABEL_40;
  v73 = 0;
  while ( 1 )
  {
    v23 = *(_BYTE *)(v4 + 16);
    v24 = v4;
    if ( v23 > 0x10u )
    {
      if ( v23 <= 0x17u )
        goto LABEL_40;
      v35 = sub_146A580(v4, a2, (__int64)&v77, v67, *(_QWORD *)(a1 + 40));
      v24 = v35;
      if ( !v35 )
        goto LABEL_40;
      v23 = *(_BYTE *)(v35 + 16);
    }
    if ( v23 != 13 )
      goto LABEL_40;
    if ( sub_13A38F0(v24 + 24, (_QWORD *)a4) )
      break;
    v81 = 0;
    v85 = &v87;
    v86 = 0x800000000LL;
    v82 = 0;
    v83 = 0;
    v84 = 0;
    if ( !(_DWORD)v79 )
      goto LABEL_36;
    v36 = v78;
    v37 = &v78[2 * v80];
    if ( v78 == v37 )
      goto LABEL_36;
    while ( 1 )
    {
      v38 = *v36;
      v39 = v36;
      if ( *v36 != -8 && v38 != -16 )
        break;
      v36 += 2;
      if ( v37 == v36 )
        goto LABEL_36;
    }
    if ( v36 == v37 )
    {
LABEL_36:
      v25 = &v87;
      v26 = 0;
      v27 = 0;
      goto LABEL_37;
    }
    v40 = 0;
    do
    {
      if ( *(_BYTE *)(v38 + 16) == 77 && v68 == *(_QWORD *)(v38 + 40) )
      {
        if ( HIDWORD(v86) <= (unsigned int)v40 )
        {
          v65 = v38;
          sub_16CD150(&v85, &v87, 0, 8);
          v40 = (unsigned int)v86;
          v38 = v65;
        }
        v85[v40] = v38;
        v40 = (unsigned int)(v86 + 1);
        LODWORD(v86) = v86 + 1;
      }
      v39 += 2;
      if ( v39 == v37 )
        break;
      while ( 1 )
      {
        v38 = *v39;
        if ( *v39 != -16 && v38 != -8 )
          break;
        v39 += 2;
        if ( v37 == v39 )
          goto LABEL_65;
      }
    }
    while ( v39 != v37 );
LABEL_65:
    v41 = v85;
    v26 = v84;
    v25 = &v85[v40];
    v27 = v82;
    if ( v25 == v85 )
      goto LABEL_37;
    v66 = v4;
    v42 = v25;
    do
    {
      v47 = *v41;
      v75 = *v41;
      if ( !v26 )
      {
        ++v81;
        goto LABEL_73;
      }
      v43 = (v26 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
      v44 = &v27[2 * v43];
      v45 = *v44;
      if ( v47 != *v44 )
      {
        v59 = 1;
        v60 = 0;
        while ( v45 != -8 )
        {
          if ( v60 || v45 != -16 )
            v44 = v60;
          v43 = (v26 - 1) & (v59 + v43);
          v62 = &v27[2 * v43];
          v45 = *v62;
          if ( v47 == *v62 )
          {
            v46 = v62[1];
            v44 = v62;
            goto LABEL_69;
          }
          ++v59;
          v60 = v44;
          v44 = &v27[2 * v43];
        }
        if ( v60 )
          v44 = v60;
        ++v81;
        v49 = v83 + 1;
        if ( 4 * ((int)v83 + 1) >= 3 * v26 )
        {
LABEL_73:
          v26 *= 2;
        }
        else
        {
          v48 = v47;
          if ( v26 - (v49 + HIDWORD(v83)) > v26 >> 3 )
            goto LABEL_75;
        }
        sub_146A3C0((__int64)&v81, v26);
        sub_1463C30((__int64)&v81, &v75, &v76);
        v44 = v76;
        v48 = v75;
        v49 = v83 + 1;
LABEL_75:
        LODWORD(v83) = v49;
        if ( *v44 != -8 )
          --HIDWORD(v83);
        *v44 = v48;
        v44[1] = 0;
LABEL_78:
        v50 = 0x17FFFFFFE8LL;
        v51 = *(_BYTE *)(v47 + 23) & 0x40;
        v52 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
        if ( v52 )
        {
          v53 = 24LL * *(unsigned int *)(v47 + 56) + 8;
          v54 = 0;
          do
          {
            v55 = v47 - 24LL * v52;
            if ( v51 )
              v55 = *(_QWORD *)(v47 - 8);
            if ( v7 == *(_QWORD *)(v55 + v53) )
            {
              v50 = 24 * v54;
              goto LABEL_85;
            }
            ++v54;
            v53 += 8;
          }
          while ( v52 != (_DWORD)v54 );
          v50 = 0x17FFFFFFE8LL;
        }
LABEL_85:
        if ( v51 )
          v56 = *(_QWORD *)(v47 - 8);
        else
          v56 = v47 - 24LL * v52;
        v57 = *(_QWORD *)(v56 + v50);
        v58 = *(_BYTE *)(v57 + 16);
        if ( v58 > 0x10u )
        {
          if ( v58 <= 0x17u )
            v57 = 0;
          else
            v57 = sub_146A580(v57, a2, (__int64)&v77, v67, *(_QWORD *)(a1 + 40));
        }
        v44[1] = v57;
        goto LABEL_70;
      }
      v46 = v44[1];
LABEL_69:
      if ( !v46 )
        goto LABEL_78;
LABEL_70:
      ++v41;
      v27 = v82;
      v26 = v84;
    }
    while ( v42 != v41 );
    v4 = v66;
    v25 = v85;
LABEL_37:
    v28 = v78;
    v78 = v27;
    v29 = v83;
    ++v77;
    LODWORD(v83) = v79;
    LODWORD(v79) = v29;
    v30 = HIDWORD(v83);
    HIDWORD(v83) = HIDWORD(v79);
    v31 = v80;
    ++v81;
    v82 = v28;
    HIDWORD(v79) = v30;
    v80 = v26;
    v84 = v31;
    if ( v25 != &v87 )
    {
      _libc_free((unsigned __int64)v25);
      v28 = v82;
    }
    j___libc_free_0(v28);
    if ( ++v73 == v70 )
    {
LABEL_40:
      v32 = sub_1456E90(a1);
      goto LABEL_41;
    }
  }
  v63 = sub_15E0530(*(_QWORD *)(a1 + 24));
  v64 = sub_1643350(v63);
  v32 = sub_145CF80(a1, v64, v73, 0);
LABEL_41:
  j___libc_free_0(v78);
  return v32;
}
