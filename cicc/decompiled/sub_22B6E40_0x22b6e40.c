// Function: sub_22B6E40
// Address: 0x22b6e40
//
__int64 __fastcall sub_22B6E40(__int64 a1, int a2, int a3)
{
  unsigned int v5; // eax
  __int64 v6; // r9
  unsigned int v7; // esi
  int v8; // eax
  int *v9; // r11
  int v10; // edx
  __int64 v11; // r8
  unsigned int v12; // r13d
  __int64 v13; // rbx
  unsigned int *v14; // rdi
  unsigned int v15; // ecx
  __int64 v16; // rax
  unsigned int *v17; // rsi
  int v18; // edx
  int v19; // r9d
  unsigned int v20; // r10d
  unsigned int *v21; // r11
  unsigned int v22; // eax
  unsigned int *v23; // rcx
  unsigned int v24; // r8d
  _DWORD *v26; // rdx
  _DWORD *v27; // rax
  unsigned int v28; // edx
  int *v29; // rdi
  int v30; // ecx
  int v31; // r12d
  unsigned int v32; // edx
  int v33; // ecx
  int v34; // r10d
  int v35; // r11d
  int v36; // r14d
  unsigned int v37; // edx
  int v38; // r13d
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  _DWORD *v41; // rax
  __int64 v42; // rdx
  _DWORD *i; // rdx
  int v44[4]; // [rsp+Ch] [rbp-B4h] BYREF
  int v45; // [rsp+1Ch] [rbp-A4h] BYREF
  __int64 v46; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v47; // [rsp+28h] [rbp-98h]
  __int64 v48; // [rsp+30h] [rbp-90h]
  unsigned int v49; // [rsp+38h] [rbp-88h]
  int v50; // [rsp+40h] [rbp-80h] BYREF
  __int64 v51; // [rsp+48h] [rbp-78h] BYREF
  __int64 v52; // [rsp+50h] [rbp-70h]
  __int64 v53; // [rsp+58h] [rbp-68h]
  unsigned int v54; // [rsp+60h] [rbp-60h]
  _QWORD v55[4]; // [rsp+70h] [rbp-50h] BYREF
  unsigned __int8 v56; // [rsp+90h] [rbp-30h]

  v44[0] = a3;
  v45 = a3;
  v46 = 0;
  v5 = sub_AF1560(2u);
  v49 = v5;
  if ( !v5 )
  {
    v47 = 0;
    v6 = 1;
    v48 = 0;
LABEL_3:
    v55[0] = 0;
    v7 = 0;
    v46 = v6;
LABEL_4:
    sub_A08C50((__int64)&v46, 2 * v7);
    goto LABEL_5;
  }
  v48 = 0;
  v11 = sub_C7D670(4LL * v5, 4);
  v47 = v11;
  v26 = (_DWORD *)(v11 + 4LL * v49);
  v7 = v49;
  if ( (_DWORD *)v11 != v26 )
  {
    v27 = (_DWORD *)v11;
    do
    {
      if ( v27 )
        *v27 = -1;
      ++v27;
    }
    while ( v26 != v27 );
  }
  v6 = v46 + 1;
  if ( !v7 )
    goto LABEL_3;
  v8 = v45;
  v28 = (v7 - 1) & (37 * v45);
  v29 = (int *)(v11 + 4LL * v28);
  v30 = *v29;
  if ( v45 == *v29 )
    goto LABEL_9;
  v31 = 1;
  v9 = 0;
  while ( v30 != -1 )
  {
    if ( v30 != -2 || v9 )
      v29 = v9;
    v28 = (v7 - 1) & (v31 + v28);
    v30 = *(_DWORD *)(v11 + 4LL * v28);
    if ( v45 == v30 )
      goto LABEL_9;
    ++v31;
    v9 = v29;
    v29 = (int *)(v11 + 4LL * v28);
  }
  ++v46;
  if ( !v9 )
    v9 = v29;
  v10 = v48 + 1;
  v55[0] = v9;
  if ( 4 * ((int)v48 + 1) >= 3 * v7 )
    goto LABEL_4;
  if ( v7 - (v10 + HIDWORD(v48)) > v7 >> 3 )
    goto LABEL_6;
  sub_A08C50((__int64)&v46, v7);
LABEL_5:
  sub_22B31A0((__int64)&v46, &v45, v55);
  v8 = v45;
  v9 = (int *)v55[0];
  v10 = v48 + 1;
LABEL_6:
  LODWORD(v48) = v10;
  if ( *v9 != -1 )
    --HIDWORD(v48);
  *v9 = v8;
  v11 = v47;
  v7 = v49;
  v6 = v46 + 1;
LABEL_9:
  v54 = v7;
  v50 = a2;
  v46 = v6;
  v52 = v11;
  v53 = v48;
  v51 = 1;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  sub_22B3900((__int64)v55, a1, &v50, (__int64)&v51);
  v12 = v56;
  v13 = v55[2];
  sub_C7D6A0(v52, 4LL * v54, 4);
  sub_C7D6A0(v47, 4LL * v49, 4);
  if ( (_BYTE)v12 )
    return v12;
  v14 = *(unsigned int **)(v13 + 16);
  v15 = *(_DWORD *)(v13 + 24);
  v16 = *(unsigned int *)(v13 + 32);
  v17 = &v14[v16];
  if ( v15 <= 1 )
  {
LABEL_31:
    if ( !(_DWORD)v16 )
      goto LABEL_33;
    v18 = v44[0];
    v19 = v16 - 1;
    goto LABEL_14;
  }
  if ( !(_DWORD)v16 )
    goto LABEL_33;
  v18 = v44[0];
  v19 = v16 - 1;
  v20 = (v16 - 1) & (37 * v44[0]);
  v21 = &v14[v20];
  v12 = *v21;
  if ( *v21 != v44[0] )
  {
    v35 = 1;
    while ( v12 != -1 )
    {
      v36 = v35 + 1;
      v20 = v19 & (v35 + v20);
      v21 = &v14[v20];
      v12 = *v21;
      if ( v44[0] == *v21 )
        goto LABEL_13;
      v35 = v36;
    }
    goto LABEL_31;
  }
LABEL_13:
  if ( v17 == v21 )
  {
LABEL_14:
    v22 = v19 & (37 * v18);
    v23 = &v14[v22];
    v24 = *v23;
    if ( *v23 == v18 )
    {
LABEL_15:
      LOBYTE(v12) = v23 != v17;
      return v12;
    }
    v33 = 1;
    while ( v24 != -1 )
    {
      v34 = v33 + 1;
      v22 = v19 & (v33 + v22);
      v23 = &v14[v22];
      v24 = *v23;
      if ( v18 == *v23 )
        goto LABEL_15;
      v33 = v34;
    }
LABEL_33:
    v23 = v17;
    goto LABEL_15;
  }
  v32 = 4 * v15;
  ++*(_QWORD *)(v13 + 8);
  if ( 4 * v15 < 0x40 )
    v32 = 64;
  if ( v32 < (unsigned int)v16 )
  {
    _BitScanReverse(&v37, v15 - 1);
    v38 = 1 << (33 - (v37 ^ 0x1F));
    if ( v38 < 64 )
      v38 = 64;
    if ( v38 == (_DWORD)v16 )
    {
      for ( *(_QWORD *)(v13 + 24) = 0; v17 != v14; ++v14 )
      {
        if ( v14 )
          *v14 = -1;
      }
    }
    else
    {
      sub_C7D6A0((__int64)v14, 4 * v16, 4);
      v39 = ((((((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
               | (4 * v38 / 3u + 1)
               | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
             | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
             | (4 * v38 / 3u + 1)
             | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
             | (4 * v38 / 3u + 1)
             | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
           | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
           | (4 * v38 / 3u + 1)
           | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 16;
      v40 = (v39
           | (((((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
               | (4 * v38 / 3u + 1)
               | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
             | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
             | (4 * v38 / 3u + 1)
             | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
             | (4 * v38 / 3u + 1)
             | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
           | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
           | (4 * v38 / 3u + 1)
           | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(v13 + 32) = v40;
      v41 = (_DWORD *)sub_C7D670(4 * v40, 4);
      v42 = *(unsigned int *)(v13 + 32);
      *(_QWORD *)(v13 + 24) = 0;
      *(_QWORD *)(v13 + 16) = v41;
      for ( i = &v41[v42]; i != v41; ++v41 )
      {
        if ( v41 )
          *v41 = -1;
      }
    }
  }
  else
  {
    if ( v17 != v14 )
      memset(v14, 255, 4 * v16);
    *(_QWORD *)(v13 + 24) = 0;
  }
  v12 = 1;
  sub_22B6470((__int64)v55, v13 + 8, v44);
  return v12;
}
