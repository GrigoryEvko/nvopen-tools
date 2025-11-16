// Function: sub_2CF8860
// Address: 0x2cf8860
//
__int64 __fastcall sub_2CF8860(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r15
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 *v9; // rsi
  unsigned int v10; // ecx
  __int64 v11; // rax
  _QWORD *v12; // rdx
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  _QWORD *v15; // rcx
  _QWORD *v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r14
  unsigned __int64 v20; // rdi
  __int64 v21; // rbx
  __int64 v22; // rbx
  int v23; // r11d
  _QWORD *v24; // r10
  _QWORD *v25; // rdi
  _QWORD *v26; // rcx
  __int64 v27; // rax
  _QWORD *v28; // rax
  unsigned int v29; // edx
  __int64 v30; // rdi
  int v31; // ecx
  _QWORD *v32; // rax
  __int64 v33; // rbx
  unsigned int v34; // eax
  __int64 v35; // rsi
  __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // r12
  unsigned __int64 v40; // rdi
  int v42; // r11d
  unsigned int v43; // edx
  _QWORD *v44; // rdi
  int v45; // r11d
  _QWORD *v46; // r10
  int v47; // ecx
  _QWORD *v48; // rdi
  unsigned int v49; // eax
  int v50; // r11d
  int v51; // r11d
  unsigned int v52; // eax
  _QWORD *v53; // rbx
  int v54; // r11d
  __int64 v56; // [rsp+10h] [rbp-100h]
  __int64 v57; // [rsp+20h] [rbp-F0h]
  __int64 v58; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v59; // [rsp+38h] [rbp-D8h]
  __int64 v60; // [rsp+40h] [rbp-D0h]
  __int64 v61; // [rsp+48h] [rbp-C8h]
  _QWORD *v62; // [rsp+50h] [rbp-C0h] BYREF
  size_t v63; // [rsp+58h] [rbp-B8h]
  __int64 v64; // [rsp+60h] [rbp-B0h]
  _BYTE v65[40]; // [rsp+68h] [rbp-A8h] BYREF
  unsigned __int64 v66[2]; // [rsp+90h] [rbp-80h] BYREF
  _QWORD *v67; // [rsp+A0h] [rbp-70h]
  __int64 v68; // [rsp+A8h] [rbp-68h]
  __int64 v69; // [rsp+B0h] [rbp-60h]
  unsigned __int64 v70; // [rsp+B8h] [rbp-58h]
  _QWORD *v71; // [rsp+C0h] [rbp-50h]
  __int64 v72; // [rsp+C8h] [rbp-48h]
  __int64 v73; // [rsp+D0h] [rbp-40h]
  __int64 *v74; // [rsp+D8h] [rbp-38h]

  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v66[1] = 8;
  v66[0] = sub_22077B0(0x40u);
  v2 = v66[0] + 24;
  v3 = sub_22077B0(0x200u);
  v6 = *(_QWORD *)(a2 + 32);
  v70 = v66[0] + 24;
  v7 = v3 + 512;
  *(_QWORD *)(v66[0] + 24) = v3;
  v68 = v3;
  v72 = v3;
  v67 = (_QWORD *)v3;
  v71 = (_QWORD *)v3;
  v69 = v3 + 512;
  v74 = (__int64 *)v2;
  v73 = v3 + 512;
  v56 = a2 + 24;
  if ( v6 != a2 + 24 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v6 )
        {
          v62 = 0;
          BUG();
        }
        v62 = (_QWORD *)(v6 - 56);
        v57 = *(_QWORD *)(v6 + 24);
        if ( v57 != v6 + 16 )
          break;
LABEL_15:
        v6 = *(_QWORD *)(v6 + 8);
        if ( v56 == v6 )
          goto LABEL_37;
      }
      while ( 1 )
      {
        if ( !v57 )
          BUG();
        v17 = *(_QWORD *)(v57 + 32);
        if ( v17 != v57 + 24 )
          break;
LABEL_26:
        v57 = *(_QWORD *)(v57 + 8);
        if ( v6 + 16 == v57 )
          goto LABEL_15;
      }
      while ( 1 )
      {
        if ( !v17 )
          BUG();
        if ( *(_BYTE *)(v17 - 24) == 63 )
        {
          v18 = v17 - 24 + 32 * (1LL - (*(_DWORD *)(v17 - 20) & 0x7FFFFFF));
          if ( v18 != v17 - 24 )
            break;
        }
LABEL_25:
        v17 = *(_QWORD *)(v17 + 8);
        if ( v57 + 24 == v17 )
          goto LABEL_26;
      }
      while ( 1 )
      {
        v19 = *(_QWORD *)v18;
        if ( !sub_BCAC40(*(_QWORD *)(*(_QWORD *)v18 + 8LL), 64) )
          goto LABEL_24;
        if ( *(_BYTE *)v19 != 17 )
          break;
        v8 = *(_DWORD *)(v19 + 32);
        v9 = *(__int64 **)(v19 + 24);
        v10 = v8 - 1;
        v7 = 1LL << ((unsigned __int8)v8 - 1);
        if ( v8 > 0x40 )
        {
          v11 = *v9;
          v7 &= v9[v10 >> 6];
          if ( v7 )
            goto LABEL_8;
LABEL_32:
          if ( v11 >= 0x80000000LL )
            goto LABEL_9;
          goto LABEL_24;
        }
        if ( (v7 & (unsigned __int64)v9) == 0 )
        {
          if ( !v8 )
            goto LABEL_24;
          v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v8)) >> (64 - (unsigned __int8)v8);
          goto LABEL_32;
        }
        if ( v8 )
        {
          v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v8)) >> (64 - (unsigned __int8)v8);
LABEL_8:
          if ( v11 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_9;
        }
LABEL_24:
        v18 += 32;
        if ( v18 == v17 - 24 )
          goto LABEL_25;
      }
      if ( *(_BYTE *)v19 == 69 && sub_2CF5CA0(v19) )
        goto LABEL_24;
LABEL_9:
      if ( !(_DWORD)v61 )
      {
        ++v58;
LABEL_102:
        sub_A35F10((__int64)&v58, 2 * v61);
        if ( (_DWORD)v61 )
        {
          v48 = v62;
          v5 = v59;
          v47 = v60 + 1;
          v49 = (v61 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v46 = (_QWORD *)(v59 + 8LL * v49);
          v12 = (_QWORD *)*v46;
          if ( v62 == (_QWORD *)*v46 )
            goto LABEL_98;
          v50 = 1;
          v4 = 0;
          while ( v12 != (_QWORD *)-4096LL )
          {
            if ( !v4 && v12 == (_QWORD *)-8192LL )
              v4 = (__int64)v46;
            v49 = (v61 - 1) & (v50 + v49);
            v46 = (_QWORD *)(v59 + 8LL * v49);
            v12 = (_QWORD *)*v46;
            if ( v62 == (_QWORD *)*v46 )
              goto LABEL_98;
            ++v50;
          }
LABEL_114:
          v12 = v48;
          if ( v4 )
            v46 = (_QWORD *)v4;
          goto LABEL_98;
        }
LABEL_140:
        LODWORD(v60) = v60 + 1;
        BUG();
      }
      v12 = v62;
      v5 = (unsigned int)(v61 - 1);
      v4 = v59;
      v13 = v5 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v14 = (_QWORD *)(v59 + 8LL * v13);
      v15 = (_QWORD *)*v14;
      if ( v62 == (_QWORD *)*v14 )
        goto LABEL_11;
      v45 = 1;
      v46 = 0;
      while ( v15 != (_QWORD *)-4096LL )
      {
        if ( v46 || v15 != (_QWORD *)-8192LL )
          v14 = v46;
        v13 = v5 & (v45 + v13);
        v53 = (_QWORD *)(v59 + 8LL * v13);
        v15 = (_QWORD *)*v53;
        if ( v62 == (_QWORD *)*v53 )
          goto LABEL_11;
        ++v45;
        v46 = v14;
        v14 = (_QWORD *)(v59 + 8LL * v13);
      }
      if ( !v46 )
        v46 = v14;
      ++v58;
      v47 = v60 + 1;
      if ( 4 * ((int)v60 + 1) >= (unsigned int)(3 * v61) )
        goto LABEL_102;
      if ( (int)v61 - HIDWORD(v60) - v47 <= (unsigned int)v61 >> 3 )
      {
        sub_A35F10((__int64)&v58, v61);
        if ( (_DWORD)v61 )
        {
          v48 = v62;
          v4 = 0;
          v5 = v59;
          v51 = 1;
          v47 = v60 + 1;
          v52 = (v61 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v46 = (_QWORD *)(v59 + 8LL * v52);
          v12 = (_QWORD *)*v46;
          if ( v62 == (_QWORD *)*v46 )
            goto LABEL_98;
          while ( v12 != (_QWORD *)-4096LL )
          {
            if ( !v4 && v12 == (_QWORD *)-8192LL )
              v4 = (__int64)v46;
            v52 = (v61 - 1) & (v51 + v52);
            v46 = (_QWORD *)(v59 + 8LL * v52);
            v12 = (_QWORD *)*v46;
            if ( v62 == (_QWORD *)*v46 )
              goto LABEL_98;
            ++v51;
          }
          goto LABEL_114;
        }
        goto LABEL_140;
      }
LABEL_98:
      LODWORD(v60) = v47;
      if ( *v46 != -4096 )
        --HIDWORD(v60);
      *v46 = v12;
LABEL_11:
      v16 = v71;
      v7 = v73 - 8;
      if ( v71 != (_QWORD *)(v73 - 8) )
      {
        if ( v71 )
        {
          v7 = (__int64)v62;
          *v71 = v62;
          v16 = v71;
        }
        v71 = v16 + 1;
        goto LABEL_15;
      }
      sub_2CBB610(v66, &v62);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v56 != v6 );
LABEL_37:
    v20 = (unsigned __int64)v71;
    if ( v71 != v67 )
    {
      while ( 1 )
      {
        if ( v20 == v72 )
        {
          v21 = *(_QWORD *)(*(v74 - 1) + 504);
          j_j___libc_free_0(v20);
          v7 = *--v74 + 512;
          v72 = *v74;
          v73 = v7;
          v71 = (_QWORD *)(v72 + 504);
        }
        else
        {
          v21 = *(_QWORD *)(v20 - 8);
          v71 = (_QWORD *)(v20 - 8);
        }
        v22 = *(_QWORD *)(v21 + 16);
        if ( v22 )
          break;
LABEL_55:
        v20 = (unsigned __int64)v71;
        if ( v71 == v67 )
          goto LABEL_56;
      }
      while ( 1 )
      {
        v27 = *(_QWORD *)(v22 + 24);
        if ( *(_BYTE *)v27 <= 0x1Cu )
          goto LABEL_43;
        v28 = *(_QWORD **)(*(_QWORD *)(v27 + 40) + 72LL);
        v62 = v28;
        if ( !(_DWORD)v61 )
          break;
        v5 = (unsigned int)(v61 - 1);
        v4 = v59;
        v23 = 1;
        v24 = 0;
        v7 = (unsigned int)v5 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v25 = (_QWORD *)(v59 + 8 * v7);
        v26 = (_QWORD *)*v25;
        if ( v28 == (_QWORD *)*v25 )
          goto LABEL_43;
        while ( v26 != (_QWORD *)-4096LL )
        {
          if ( v24 || v26 != (_QWORD *)-8192LL )
            v25 = v24;
          v7 = (unsigned int)v5 & (v23 + (_DWORD)v7);
          v26 = *(_QWORD **)(v59 + 8LL * (unsigned int)v7);
          if ( v28 == v26 )
            goto LABEL_43;
          ++v23;
          v24 = v25;
          v25 = (_QWORD *)(v59 + 8LL * (unsigned int)v7);
        }
        if ( !v24 )
          v24 = v25;
        ++v58;
        v31 = v60 + 1;
        if ( 4 * ((int)v60 + 1) >= (unsigned int)(3 * v61) )
          goto LABEL_47;
        if ( (int)v61 - HIDWORD(v60) - v31 <= (unsigned int)v61 >> 3 )
        {
          sub_A35F10((__int64)&v58, v61);
          if ( !(_DWORD)v61 )
          {
LABEL_144:
            LODWORD(v60) = v60 + 1;
            BUG();
          }
          v28 = v62;
          v4 = v59;
          v5 = 0;
          v42 = 1;
          v43 = (v61 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v24 = (_QWORD *)(v59 + 8LL * v43);
          v44 = (_QWORD *)*v24;
          v31 = v60 + 1;
          if ( (_QWORD *)*v24 != v62 )
          {
            while ( v44 != (_QWORD *)-4096LL )
            {
              if ( v44 == (_QWORD *)-8192LL && !v5 )
                v5 = (__int64)v24;
              v43 = (v61 - 1) & (v42 + v43);
              v24 = (_QWORD *)(v59 + 8LL * v43);
              v44 = (_QWORD *)*v24;
              if ( v62 == (_QWORD *)*v24 )
                goto LABEL_49;
              ++v42;
            }
            goto LABEL_83;
          }
        }
LABEL_49:
        LODWORD(v60) = v31;
        if ( *v24 != -4096 )
          --HIDWORD(v60);
        *v24 = v28;
        v32 = v71;
        v7 = v73 - 8;
        if ( v71 == (_QWORD *)(v73 - 8) )
        {
          sub_2CBB610(v66, &v62);
LABEL_43:
          v22 = *(_QWORD *)(v22 + 8);
          if ( !v22 )
            goto LABEL_55;
          continue;
        }
        if ( v71 )
        {
          v7 = (__int64)v62;
          *v71 = v62;
          v32 = v71;
        }
        v71 = v32 + 1;
        v22 = *(_QWORD *)(v22 + 8);
        if ( !v22 )
          goto LABEL_55;
      }
      ++v58;
LABEL_47:
      sub_A35F10((__int64)&v58, 2 * v61);
      if ( !(_DWORD)v61 )
        goto LABEL_144;
      v28 = v62;
      v4 = v59;
      v29 = (v61 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v24 = (_QWORD *)(v59 + 8LL * v29);
      v30 = *v24;
      v31 = v60 + 1;
      if ( (_QWORD *)*v24 != v62 )
      {
        v54 = 1;
        v5 = 0;
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v5 )
            v5 = (__int64)v24;
          v29 = (v61 - 1) & (v54 + v29);
          v24 = (_QWORD *)(v59 + 8LL * v29);
          v30 = *v24;
          if ( v62 == (_QWORD *)*v24 )
            goto LABEL_49;
          ++v54;
        }
LABEL_83:
        if ( v5 )
          v24 = (_QWORD *)v5;
        goto LABEL_49;
      }
      goto LABEL_49;
    }
LABEL_56:
    v33 = *(_QWORD *)(a2 + 32);
    if ( v56 == v33 )
      goto LABEL_65;
    while ( 1 )
    {
      v36 = v33 - 56;
      if ( !v33 )
        v36 = 0;
      if ( !(_DWORD)v61 )
        goto LABEL_63;
      v7 = (unsigned int)(v61 - 1);
      v34 = v7 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v35 = *(_QWORD *)(v59 + 8LL * v34);
      if ( v36 == v35 )
      {
LABEL_59:
        v33 = *(_QWORD *)(v33 + 8);
        if ( v56 == v33 )
          break;
      }
      else
      {
        v4 = 1;
        while ( v35 != -4096 )
        {
          v5 = (unsigned int)(v4 + 1);
          v34 = v7 & (v4 + v34);
          v35 = *(_QWORD *)(v59 + 8LL * v34);
          if ( v36 == v35 )
            goto LABEL_59;
          v4 = (unsigned int)v5;
        }
LABEL_63:
        v63 = 0;
        v62 = v65;
        v64 = 32;
        sub_2CF5D20(v36, &v62, v7, v59, v4, v5);
        sub_BA8E40(a2, v62, v63);
        if ( v62 == (_QWORD *)v65 )
          goto LABEL_59;
        _libc_free((unsigned __int64)v62);
        v33 = *(_QWORD *)(v33 + 8);
        if ( v56 == v33 )
          break;
      }
    }
  }
LABEL_65:
  v37 = v66[0];
  if ( v66[0] )
  {
    v38 = (unsigned __int64 *)v70;
    v39 = (unsigned __int64 *)(v74 + 1);
    if ( (unsigned __int64)(v74 + 1) > v70 )
    {
      do
      {
        v40 = *v38++;
        j_j___libc_free_0(v40);
      }
      while ( v39 > v38 );
      v37 = v66[0];
    }
    j_j___libc_free_0(v37);
  }
  return sub_C7D6A0(v59, 8LL * (unsigned int)v61, 8);
}
