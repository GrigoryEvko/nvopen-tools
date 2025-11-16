// Function: sub_2EC7BC0
// Address: 0x2ec7bc0
//
void __fastcall sub_2EC7BC0(__int64 a1, unsigned __int64 **a2, char a3)
{
  unsigned __int64 *v6; // r15
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rcx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int64 *v22; // rdi
  unsigned __int64 v23; // rax
  __int64 v24; // rcx
  unsigned __int64 i; // r8
  __int16 v26; // dx
  __int64 v27; // rdi
  __int64 v28; // rcx
  unsigned int v29; // r11d
  __int64 *v30; // rdx
  __int64 v31; // r8
  __int64 v32; // rsi
  unsigned __int64 *v33; // rdi
  unsigned __int64 v34; // rax
  __int64 v35; // rcx
  unsigned __int64 j; // r8
  __int16 v37; // dx
  __int64 v38; // rdi
  __int64 v39; // rcx
  unsigned int v40; // r10d
  __int64 *v41; // rdx
  __int64 v42; // r8
  __int64 v43; // rdi
  __int64 v44; // rdi
  __int64 v45; // rax
  int v46; // edx
  int v47; // r11d
  int v48; // edx
  int v49; // r9d
  _BYTE *v50; // [rsp+20h] [rbp-370h] BYREF
  __int64 v51; // [rsp+28h] [rbp-368h]
  _BYTE v52[192]; // [rsp+30h] [rbp-360h] BYREF
  _BYTE *v53; // [rsp+F0h] [rbp-2A0h] BYREF
  __int64 v54; // [rsp+F8h] [rbp-298h]
  _BYTE v55[192]; // [rsp+100h] [rbp-290h] BYREF
  _BYTE *v56; // [rsp+1C0h] [rbp-1D0h]
  __int64 v57; // [rsp+1C8h] [rbp-1C8h]
  _BYTE v58[192]; // [rsp+1D0h] [rbp-1C0h] BYREF
  _BYTE *v59; // [rsp+290h] [rbp-100h]
  __int64 v60; // [rsp+298h] [rbp-F8h]
  _BYTE v61[240]; // [rsp+2A0h] [rbp-F0h] BYREF

  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 3504);
  if ( !a3 )
  {
    v16 = sub_2EC1A40(*(_QWORD *)(a1 + 3512), v7);
    if ( (unsigned __int64 *)v16 == v6 )
    {
      *(_QWORD *)(a1 + 3512) = v6;
    }
    else
    {
      v17 = *(_QWORD *)(a1 + 3504);
      if ( v6 == (unsigned __int64 *)v17 )
      {
        if ( !v6 )
          BUG();
        if ( (*(_BYTE *)v6 & 4) == 0 && (*((_BYTE *)v6 + 44) & 8) != 0 )
        {
          do
            v17 = *(_QWORD *)(v17 + 8);
          while ( (*(_BYTE *)(v17 + 44) & 8) != 0 );
        }
        v44 = *(_QWORD *)(v17 + 8);
        *(_QWORD *)(a1 + 3504) = v44;
        v45 = sub_2EC2050(v44, v16);
        *(_QWORD *)(a1 + 3504) = v45;
        *(_QWORD *)(a1 + 5440) = v45;
      }
      sub_2EC62F0((_QWORD *)a1, v6, *(unsigned __int64 **)(a1 + 3512));
      *(_QWORD *)(a1 + 3512) = v6;
      *(_QWORD *)(a1 + 6312) = v6;
    }
    if ( !*(_BYTE *)(a1 + 4016) )
      return;
    v18 = *(_QWORD *)(a1 + 24);
    v19 = *(unsigned __int8 *)(a1 + 4017);
    v56 = v58;
    v59 = v61;
    v20 = *(_QWORD *)(a1 + 40);
    v53 = v55;
    v54 = 0x800000000LL;
    v57 = 0x800000000LL;
    v60 = 0x800000000LL;
    sub_2F75980(&v53, v6, v18, v20, v19, 0);
    if ( !*(_BYTE *)(a1 + 4017) )
    {
      sub_2F761E0(&v53, v6, *(_QWORD *)(a1 + 3464));
LABEL_47:
      v43 = a1 + 6248;
      if ( *(_QWORD *)(a1 + 6312) != *(_QWORD *)(a1 + 3512) )
      {
        sub_2F771D0();
        v43 = a1 + 6248;
      }
      v50 = v52;
      v51 = 0x800000000LL;
      sub_2F78160(v43, &v53, &v50);
      sub_2EC6C00((_QWORD *)a1, (__int64)a2, *(_QWORD **)(a1 + 6296));
      sub_2EC6D00(a1, (__int64)v50, (unsigned int)v51);
      if ( v50 != v52 )
        _libc_free((unsigned __int64)v50);
      v15 = (unsigned __int64)v59;
      if ( v59 == v61 )
        goto LABEL_14;
      goto LABEL_13;
    }
    v21 = *(_QWORD *)(a1 + 3464);
    v22 = v6;
    v23 = (unsigned __int64)v6;
    v24 = *(_QWORD *)(v21 + 32);
    if ( (*((_DWORD *)v6 + 11) & 4) != 0 )
    {
      do
        v23 = *(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v23 + 44) & 4) != 0 );
    }
    if ( (*((_DWORD *)v6 + 11) & 8) != 0 )
    {
      do
        v22 = (unsigned __int64 *)v22[1];
      while ( (*((_BYTE *)v22 + 44) & 8) != 0 );
    }
    for ( i = v22[1]; i != v23; v23 = *(_QWORD *)(v23 + 8) )
    {
      v26 = *(_WORD *)(v23 + 68);
      if ( (unsigned __int16)(v26 - 14) > 4u && v26 != 24 )
        break;
    }
    v27 = *(_QWORD *)(v24 + 128);
    v28 = *(unsigned int *)(v24 + 144);
    if ( (_DWORD)v28 )
    {
      v29 = (v28 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v30 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v30;
      if ( v23 == *v30 )
      {
LABEL_33:
        sub_2F76630(&v53, v21, *(_QWORD *)(a1 + 40), v30[1] & 0xFFFFFFFFFFFFFFF8LL | 4, v6);
        goto LABEL_47;
      }
      v48 = 1;
      while ( v31 != -4096 )
      {
        v49 = v48 + 1;
        v29 = (v28 - 1) & (v48 + v29);
        v30 = (__int64 *)(v27 + 16LL * v29);
        v31 = *v30;
        if ( v23 == *v30 )
          goto LABEL_33;
        v48 = v49;
      }
    }
    v30 = (__int64 *)(v27 + 16 * v28);
    goto LABEL_33;
  }
  if ( v6 != (unsigned __int64 *)v7 )
  {
    sub_2EC62F0((_QWORD *)a1, *a2, (unsigned __int64 *)v7);
    *(_QWORD *)(a1 + 5440) = v6;
    if ( !*(_BYTE *)(a1 + 4016) )
      return;
LABEL_10:
    v12 = *(_QWORD *)(a1 + 24);
    v13 = *(unsigned __int8 *)(a1 + 4017);
    v56 = v58;
    v59 = v61;
    v14 = *(_QWORD *)(a1 + 40);
    v53 = v55;
    v54 = 0x800000000LL;
    v57 = 0x800000000LL;
    v60 = 0x800000000LL;
    sub_2F75980(&v53, v6, v12, v14, v13, 0);
    if ( !*(_BYTE *)(a1 + 4017) )
    {
      sub_2F761E0(&v53, v6, *(_QWORD *)(a1 + 3464));
      goto LABEL_12;
    }
    v32 = *(_QWORD *)(a1 + 3464);
    v33 = v6;
    v34 = (unsigned __int64)v6;
    v35 = *(_QWORD *)(v32 + 32);
    if ( (*((_DWORD *)v6 + 11) & 4) != 0 )
    {
      do
        v34 = *(_QWORD *)v34 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v34 + 44) & 4) != 0 );
    }
    if ( (*((_DWORD *)v6 + 11) & 8) != 0 )
    {
      do
        v33 = (unsigned __int64 *)v33[1];
      while ( (*((_BYTE *)v33 + 44) & 8) != 0 );
    }
    for ( j = v33[1]; j != v34; v34 = *(_QWORD *)(v34 + 8) )
    {
      v37 = *(_WORD *)(v34 + 68);
      if ( (unsigned __int16)(v37 - 14) > 4u && v37 != 24 )
        break;
    }
    v38 = *(_QWORD *)(v35 + 128);
    v39 = *(unsigned int *)(v35 + 144);
    if ( (_DWORD)v39 )
    {
      v40 = (v39 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v41 = (__int64 *)(v38 + 16LL * v40);
      v42 = *v41;
      if ( v34 == *v41 )
      {
LABEL_44:
        sub_2F76630(&v53, v32, *(_QWORD *)(a1 + 40), v41[1] & 0xFFFFFFFFFFFFFFF8LL | 4, v6);
LABEL_12:
        sub_2F77D30(a1 + 5376, &v53);
        sub_2EC6C00((_QWORD *)a1, (__int64)a2, *(_QWORD **)(a1 + 5424));
        v15 = (unsigned __int64)v59;
        if ( v59 == v61 )
        {
LABEL_14:
          if ( v56 != v58 )
            _libc_free((unsigned __int64)v56);
          if ( v53 != v55 )
            _libc_free((unsigned __int64)v53);
          return;
        }
LABEL_13:
        _libc_free(v15);
        goto LABEL_14;
      }
      v46 = 1;
      while ( v42 != -4096 )
      {
        v47 = v46 + 1;
        v40 = (v39 - 1) & (v46 + v40);
        v41 = (__int64 *)(v38 + 16LL * v40);
        v42 = *v41;
        if ( *v41 == v34 )
          goto LABEL_44;
        v46 = v47;
      }
    }
    v41 = (__int64 *)(v38 + 16 * v39);
    goto LABEL_44;
  }
  v8 = *(_QWORD *)(a1 + 3512);
  if ( !v6 )
    BUG();
  if ( (*(_BYTE *)v6 & 4) == 0 && (*((_BYTE *)v6 + 44) & 8) != 0 )
  {
    do
      v7 = *(_QWORD *)(v7 + 8);
    while ( (*(_BYTE *)(v7 + 44) & 8) != 0 );
  }
  v9 = *(_QWORD *)(v7 + 8);
  *(_QWORD *)(a1 + 3504) = v9;
  v10 = sub_2EC2050(v9, v8);
  v11 = *(_BYTE *)(a1 + 4016) == 0;
  *(_QWORD *)(a1 + 3504) = v10;
  if ( !v11 )
    goto LABEL_10;
}
