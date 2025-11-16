// Function: sub_1357740
// Address: 0x1357740
//
__int64 __fastcall sub_1357740(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // r14
  char v6; // dl
  char v7; // al
  char v8; // al
  _QWORD *v9; // rdx
  _QWORD *v10; // r9
  char *v11; // r13
  char *v12; // r8
  __int64 v13; // r15
  char *v14; // r14
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 result; // rax
  int v20; // eax
  int v21; // edx
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 *v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rax
  bool v39; // cf
  unsigned __int64 v40; // rax
  _QWORD *v41; // r15
  __int64 v42; // rsi
  char *v43; // r14
  __int64 v44; // rsi
  _QWORD *v45; // r14
  __int64 v46; // rsi
  _QWORD *v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // r15
  __int64 v50; // rax
  __int64 v51; // [rsp+0h] [rbp-C0h]
  _QWORD *v52; // [rsp+8h] [rbp-B8h]
  _QWORD *v53; // [rsp+10h] [rbp-B0h]
  char *v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  char *v56; // [rsp+18h] [rbp-A8h]
  _QWORD *v57; // [rsp+18h] [rbp-A8h]
  char *v58; // [rsp+18h] [rbp-A8h]
  char *v59; // [rsp+18h] [rbp-A8h]
  _QWORD *v60; // [rsp+18h] [rbp-A8h]
  _QWORD *v61; // [rsp+20h] [rbp-A0h]
  char *v62; // [rsp+20h] [rbp-A0h]
  _QWORD *v63; // [rsp+20h] [rbp-A0h]
  char *v64; // [rsp+20h] [rbp-A0h]
  _QWORD *v65; // [rsp+20h] [rbp-A0h]
  _QWORD *v66; // [rsp+20h] [rbp-A0h]
  char *v67; // [rsp+20h] [rbp-A0h]
  char *v68; // [rsp+20h] [rbp-A0h]
  _QWORD v70[6]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v71[12]; // [rsp+60h] [rbp-60h] BYREF

  v5 = *(_BYTE *)(a1 + 67);
  v6 = (v5 | *(_BYTE *)(a2 + 67)) & 0x30 | v5 & 0xCF;
  *(_BYTE *)(a1 + 67) = v6;
  v7 = v6 & 0xBF | (v6 | *(_BYTE *)(a2 + 67)) & 0x40;
  *(_BYTE *)(a1 + 67) = v7;
  v8 = (v7 | *(_BYTE *)(a2 + 67)) & 0x80 | v7 & 0x7F;
  *(_BYTE *)(a1 + 67) = v8;
  if ( (v8 & 0x40) != 0 )
    goto LABEL_2;
  v22 = *(__int64 **)(a2 + 16);
  v23 = v22[5];
  v24 = *(_QWORD *)a3;
  v25 = v22[6];
  v26 = *(__int64 **)(a1 + 16);
  v27 = v22[7];
  if ( (v23 == -8 || v23 == -16) && !v25 && !v27 )
    v23 = v22[6];
  v28 = v22[4];
  v29 = *v22;
  v71[2] = v23;
  v71[3] = v25;
  v30 = v26[6];
  v71[0] = v29;
  v31 = v26[5];
  v71[1] = v28;
  v32 = v26[7];
  v71[4] = v27;
  if ( (v31 == -8 || v31 == -16) && !v30 && !v32 )
    v31 = 0;
  v33 = v26[4];
  v34 = *v26;
  v70[2] = v31;
  v70[4] = v32;
  v70[0] = v34;
  v70[1] = v33;
  v70[3] = v30;
  if ( (unsigned __int8)sub_134CB50(v24, (__int64)v70, (__int64)v71) != 3 )
    *(_BYTE *)(a1 + 67) |= 0x40u;
  if ( (*(_BYTE *)(a1 + 67) & 0x40) != 0 )
  {
LABEL_2:
    if ( (v5 & 0x40) == 0 )
      *(_DWORD *)(a3 + 56) += *(_DWORD *)(a1 + 68);
    if ( (*(_BYTE *)(a2 + 67) & 0x40) == 0 )
      *(_DWORD *)(a3 + 56) += *(_DWORD *)(a2 + 68);
  }
  v9 = *(_QWORD **)(a1 + 48);
  v10 = *(_QWORD **)(a1 + 40);
  v11 = *(char **)(a2 + 48);
  v12 = *(char **)(a2 + 40);
  if ( v10 != v9 )
  {
    if ( v11 == v12 )
      goto LABEL_22;
    v13 = v11 - v12;
    if ( *(_QWORD *)(a1 + 56) - (_QWORD)v9 >= (unsigned __int64)(v11 - v12) )
    {
      v14 = *(char **)(a2 + 40);
      do
      {
        if ( v9 )
        {
          *v9 = 4;
          v9[1] = 0;
          v15 = *((_QWORD *)v14 + 2);
          v9[2] = v15;
          if ( v15 != 0 && v15 != -8 && v15 != -16 )
          {
            v54 = v12;
            v61 = v9;
            sub_1649AC0(v9, *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL);
            v12 = v54;
            v9 = v61;
          }
        }
        v14 += 24;
        v9 += 3;
      }
      while ( v11 != v14 );
      *(_QWORD *)(a1 + 48) += v13;
LABEL_16:
      v16 = *(_QWORD *)(a2 + 40);
      v17 = *(_QWORD *)(a2 + 48);
      v55 = v16;
      if ( v16 != v17 )
      {
        do
        {
          v18 = *(_QWORD *)(v16 + 16);
          if ( v18 != 0 && v18 != -8 && v18 != -16 )
          {
            v62 = v12;
            sub_1649B30(v16);
            v12 = v62;
          }
          v16 += 24;
        }
        while ( v17 != v16 );
        *(_QWORD *)(a2 + 48) = v55;
      }
      goto LABEL_22;
    }
    v36 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
    v37 = 0xAAAAAAAAAAAAAAABLL * (v9 - v10);
    if ( v36 > 0x555555555555555LL - v37 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    v38 = v36;
    if ( v36 < v37 )
      v38 = 0xAAAAAAAAAAAAAAABLL * (v9 - v10);
    v39 = __CFADD__(v37, v38);
    v40 = v37 + v38;
    if ( v39 )
    {
      v49 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v40 )
      {
        v51 = 0;
        v53 = 0;
        goto LABEL_52;
      }
      if ( v40 > 0x555555555555555LL )
        v40 = 0x555555555555555LL;
      v49 = 24 * v40;
    }
    v60 = *(_QWORD **)(a1 + 48);
    v68 = *(char **)(a2 + 40);
    v50 = sub_22077B0(v49);
    v10 = *(_QWORD **)(a1 + 40);
    v9 = v60;
    v53 = (_QWORD *)v50;
    v12 = v68;
    v51 = v50 + v49;
    v41 = (_QWORD *)v50;
    if ( v60 == v10 )
    {
LABEL_58:
      v43 = v12;
      do
      {
        if ( v41 )
        {
          *v41 = 4;
          v41[1] = 0;
          v44 = *((_QWORD *)v43 + 2);
          v41[2] = v44;
          if ( v44 != -8 && v44 != 0 && v44 != -16 )
          {
            v57 = v9;
            v64 = v12;
            sub_1649AC0(v41, *(_QWORD *)v43 & 0xFFFFFFFFFFFFFFF8LL);
            v9 = v57;
            v12 = v64;
          }
        }
        v43 += 24;
        v41 += 3;
      }
      while ( v11 != v43 );
      v45 = *(_QWORD **)(a1 + 48);
      if ( v9 != v45 )
      {
        do
        {
          *v41 = 4;
          v41[1] = 0;
          v46 = v9[2];
          v41[2] = v46;
          if ( v46 != 0 && v46 != -8 && v46 != -16 )
          {
            v58 = v12;
            v65 = v9;
            sub_1649AC0(v41, *v9 & 0xFFFFFFFFFFFFFFF8LL);
            v12 = v58;
            v9 = v65;
          }
          v9 += 3;
          v41 += 3;
        }
        while ( v45 != v9 );
        v45 = *(_QWORD **)(a1 + 48);
      }
      v47 = *(_QWORD **)(a1 + 40);
      if ( v47 != v45 )
      {
        do
        {
          v48 = v47[2];
          if ( v48 != 0 && v48 != -8 && v48 != -16 )
          {
            v59 = v12;
            v66 = v47;
            sub_1649B30(v47);
            v12 = v59;
            v47 = v66;
          }
          v47 += 3;
        }
        while ( v47 != v45 );
        v45 = *(_QWORD **)(a1 + 40);
      }
      if ( v45 )
      {
        v67 = v12;
        j_j___libc_free_0(v45, *(_QWORD *)(a1 + 56) - (_QWORD)v45);
        v12 = v67;
      }
      *(_QWORD *)(a1 + 48) = v41;
      *(_QWORD *)(a1 + 40) = v53;
      *(_QWORD *)(a1 + 56) = v51;
      goto LABEL_16;
    }
LABEL_52:
    v41 = v53;
    do
    {
      if ( v41 )
      {
        *v41 = 4;
        v41[1] = 0;
        v42 = v10[2];
        v41[2] = v42;
        if ( v42 != 0 && v42 != -8 && v42 != -16 )
        {
          v52 = v9;
          v56 = v12;
          v63 = v10;
          sub_1649AC0(v41, *v10 & 0xFFFFFFFFFFFFFFF8LL);
          v9 = v52;
          v12 = v56;
          v10 = v63;
        }
      }
      v10 += 3;
      v41 += 3;
    }
    while ( v9 != v10 );
    goto LABEL_58;
  }
  if ( v11 != v12 )
  {
    *(_QWORD *)(a1 + 40) = v12;
    v35 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
    *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 56);
    *(_QWORD *)(a2 + 40) = v10;
    *(_QWORD *)(a2 + 48) = v10;
    *(_QWORD *)(a2 + 56) = v35;
    *(_DWORD *)(a1 + 64) = (*(_DWORD *)(a1 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(a1 + 64) & 0xF8000000;
  }
LABEL_22:
  *(_QWORD *)(a2 + 32) = a1;
  result = (*(_DWORD *)(a1 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(a1 + 64) & 0xF8000000;
  *(_DWORD *)(a1 + 64) = result;
  if ( *(_QWORD *)(a2 + 16) )
  {
    *(_DWORD *)(a1 + 68) += *(_DWORD *)(a2 + 68);
    *(_DWORD *)(a2 + 68) = 0;
    **(_QWORD **)(a1 + 24) = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
    result = a2 + 16;
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 24) = a2 + 16;
  }
  if ( v11 != v12 )
  {
    v20 = *(_DWORD *)(a2 + 64);
    v21 = (v20 + 0x7FFFFFF) & 0x7FFFFFF;
    result = v21 | v20 & 0xF8000000;
    *(_DWORD *)(a2 + 64) = result;
    if ( !v21 )
      return sub_1357730(a2, a3);
  }
  return result;
}
