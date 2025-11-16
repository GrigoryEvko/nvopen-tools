// Function: sub_1DE8B00
// Address: 0x1de8b00
//
__int64 __fastcall sub_1DE8B00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r10
  __int64 *v7; // r9
  __int64 *v9; // r13
  __int64 *v10; // r12
  __int64 v11; // rsi
  __int64 *v12; // rdi
  __int64 *v13; // rax
  __int64 *v14; // rcx
  __int64 *v15; // r13
  __int64 v16; // rsi
  int v17; // ecx
  int v18; // r8d
  unsigned int v19; // eax
  __int64 v20; // rdi
  unsigned int v21; // esi
  __int64 v22; // r9
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rdx
  int v27; // ecx
  int v28; // r11d
  __int64 *v29; // rdi
  int v30; // eax
  int v31; // eax
  unsigned int v32; // edx
  char v33; // al
  unsigned int v34; // r14d
  int v36; // eax
  int v37; // r10d
  __int64 v38; // r11
  unsigned int v39; // ecx
  int v40; // r9d
  __int64 *v41; // rsi
  __int64 v42; // [rsp+8h] [rbp-B8h]
  __int64 *v46; // [rsp+28h] [rbp-98h]
  __int64 v47; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v48; // [rsp+38h] [rbp-88h] BYREF
  __int64 v49; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v50; // [rsp+48h] [rbp-78h]
  __int64 *v51; // [rsp+50h] [rbp-70h]
  __int64 v52; // [rsp+58h] [rbp-68h]
  int i; // [rsp+60h] [rbp-60h]
  _BYTE v54[88]; // [rsp+68h] [rbp-58h] BYREF

  v6 = (__int64 *)v54;
  v7 = (__int64 *)v54;
  v9 = *(__int64 **)(a2 + 96);
  v10 = *(__int64 **)(a2 + 88);
  v49 = 0;
  v50 = (__int64 *)v54;
  v51 = (__int64 *)v54;
  v52 = 4;
  for ( i = 0; v9 != v10; ++v10 )
  {
LABEL_5:
    v11 = *v10;
    if ( v7 != v6 )
      goto LABEL_3;
    v12 = &v7[HIDWORD(v52)];
    if ( v12 != v7 )
    {
      v13 = v7;
      v14 = 0;
      while ( v11 != *v13 )
      {
        if ( *v13 == -2 )
          v14 = v13;
        if ( v12 == ++v13 )
        {
          if ( !v14 )
            goto LABEL_44;
          ++v10;
          *v14 = v11;
          v7 = v51;
          --i;
          v6 = v50;
          ++v49;
          if ( v9 != v10 )
            goto LABEL_5;
          goto LABEL_14;
        }
      }
      continue;
    }
LABEL_44:
    if ( HIDWORD(v52) < (unsigned int)v52 )
    {
      ++HIDWORD(v52);
      *v12 = v11;
      v6 = v50;
      ++v49;
      v7 = v51;
    }
    else
    {
LABEL_3:
      sub_16CCBA0((__int64)&v49, v11);
      v7 = v51;
      v6 = v50;
    }
  }
LABEL_14:
  v15 = *(__int64 **)(a3 + 64);
  v42 = a1 + 888;
  v46 = *(__int64 **)(a3 + 72);
  if ( v46 != v15 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v26 = *v15;
        v47 = v26;
        if ( a2 == v26 )
          goto LABEL_21;
        if ( !a5 )
          break;
        if ( (*(_BYTE *)(a5 + 8) & 1) != 0 )
        {
          v16 = a5 + 16;
          v17 = 15;
        }
        else
        {
          v27 = *(_DWORD *)(a5 + 24);
          v16 = *(_QWORD *)(a5 + 16);
          if ( !v27 )
            goto LABEL_21;
          v17 = v27 - 1;
        }
        v18 = 1;
        v19 = v17 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v20 = *(_QWORD *)(v16 + 8LL * v19);
        if ( v26 == v20 )
          break;
        while ( v20 != -8 )
        {
          v19 = v17 & (v18 + v19);
          v20 = *(_QWORD *)(v16 + 8LL * v19);
          if ( v26 == v20 )
            goto LABEL_18;
          ++v18;
        }
        if ( v46 == ++v15 )
          goto LABEL_50;
      }
LABEL_18:
      v21 = *(_DWORD *)(a1 + 912);
      if ( !v21 )
        break;
      v22 = *(_QWORD *)(a1 + 896);
      v23 = (v21 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v24 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v24;
      if ( v26 == *v24 )
      {
LABEL_20:
        if ( v24[1] == a4 )
          goto LABEL_21;
        goto LABEL_37;
      }
      v28 = 1;
      v29 = 0;
      while ( v25 != -8 )
      {
        if ( v25 == -16 && !v29 )
          v29 = v24;
        v23 = (v21 - 1) & (v28 + v23);
        v24 = (__int64 *)(v22 + 16LL * v23);
        v25 = *v24;
        if ( v26 == *v24 )
          goto LABEL_20;
        ++v28;
      }
      if ( !v29 )
        v29 = v24;
      v30 = *(_DWORD *)(a1 + 904);
      ++*(_QWORD *)(a1 + 888);
      v31 = v30 + 1;
      if ( 4 * v31 >= 3 * v21 )
        goto LABEL_52;
      if ( v21 - *(_DWORD *)(a1 + 908) - v31 <= v21 >> 3 )
      {
        sub_1DE4DF0(v42, v21);
        sub_1DE30F0(v42, &v47, &v48);
        v29 = v48;
        v26 = v47;
        v31 = *(_DWORD *)(a1 + 904) + 1;
      }
LABEL_34:
      *(_DWORD *)(a1 + 904) = v31;
      if ( *v29 != -8 )
        --*(_DWORD *)(a1 + 908);
      v29[1] = 0;
      *v29 = v26;
      v26 = v47;
LABEL_37:
      v32 = sub_1F350E0(a1 + 616, a3, v26);
      if ( !(_BYTE)v32 )
      {
        if ( (unsigned int)(HIDWORD(v52) - i) <= 1 || (v33 = sub_1DE89B0(v47, (__int64)&v49), v32 = 0, !v33) )
        {
          v34 = v32;
          goto LABEL_41;
        }
      }
LABEL_21:
      if ( v46 == ++v15 )
        goto LABEL_50;
    }
    ++*(_QWORD *)(a1 + 888);
LABEL_52:
    sub_1DE4DF0(v42, 2 * v21);
    v36 = *(_DWORD *)(a1 + 912);
    if ( !v36 )
    {
      ++*(_DWORD *)(a1 + 904);
      BUG();
    }
    v37 = v36 - 1;
    v38 = *(_QWORD *)(a1 + 896);
    v39 = (v36 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v31 = *(_DWORD *)(a1 + 904) + 1;
    v29 = (__int64 *)(v38 + 16LL * v39);
    v26 = *v29;
    if ( v47 != *v29 )
    {
      v40 = 1;
      v41 = 0;
      while ( v26 != -8 )
      {
        if ( !v41 && v26 == -16 )
          v41 = v29;
        v39 = v37 & (v40 + v39);
        v29 = (__int64 *)(v38 + 16LL * v39);
        v26 = *v29;
        if ( v47 == *v29 )
          goto LABEL_34;
        ++v40;
      }
      v26 = v47;
      if ( v41 )
        v29 = v41;
    }
    goto LABEL_34;
  }
LABEL_50:
  v34 = 1;
LABEL_41:
  if ( v51 != v50 )
    _libc_free((unsigned __int64)v51);
  return v34;
}
