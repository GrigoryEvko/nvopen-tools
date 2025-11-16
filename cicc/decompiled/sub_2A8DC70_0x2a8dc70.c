// Function: sub_2A8DC70
// Address: 0x2a8dc70
//
void __fastcall sub_2A8DC70(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned int v10; // r12d
  __int64 v11; // rsi
  __int64 v12; // rcx
  char v13; // al
  char v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // r11
  int v23; // r13d
  unsigned int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // edi
  __int64 *v27; // rdx
  __int64 v28; // rcx
  unsigned int v29; // ecx
  unsigned int v30; // eax
  _QWORD *v31; // rdi
  int v32; // r12d
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *j; // rdx
  __int64 v38; // rax
  char v39; // al
  __int64 v40; // r13
  char v41; // di
  _QWORD *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // r13
  __int64 v46; // rdx
  unsigned int v47; // esi
  int v48; // r12d
  int v49; // r12d
  __int64 v50; // r10
  unsigned int v51; // ecx
  int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // rdi
  int v55; // r9d
  __int64 *v56; // rsi
  int v57; // r10d
  int v58; // ecx
  int v59; // eax
  int v60; // r10d
  __int64 v61; // r9
  int v62; // r8d
  unsigned int v63; // r12d
  __int64 *v64; // rcx
  __int64 v65; // rsi
  _QWORD *v66; // rax
  __int64 v67; // [rsp+0h] [rbp-D0h]
  __int64 v68; // [rsp+8h] [rbp-C8h]
  __int64 v69; // [rsp+10h] [rbp-C0h]
  __int64 v70; // [rsp+10h] [rbp-C0h]
  __int64 v71; // [rsp+10h] [rbp-C0h]
  __int64 v72; // [rsp+20h] [rbp-B0h]
  __int64 v73; // [rsp+20h] [rbp-B0h]
  __int64 v74; // [rsp+20h] [rbp-B0h]
  _QWORD *v75; // [rsp+30h] [rbp-A0h]
  __int64 v76; // [rsp+38h] [rbp-98h]
  char v77[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v78; // [rsp+60h] [rbp-70h]
  _BYTE v79[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v80; // [rsp+90h] [rbp-40h]

  v67 = a1 + 1304;
  v3 = *(_DWORD *)(a1 + 1320);
  ++*(_QWORD *)(a1 + 1304);
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 1324) )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 1328);
    if ( (unsigned int)v4 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1312), 24 * v4, 8);
      *(_QWORD *)(a1 + 1312) = 0;
      *(_QWORD *)(a1 + 1320) = 0;
      *(_DWORD *)(a1 + 1328) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v29 = 4 * v3;
  v4 = *(unsigned int *)(a1 + 1328);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v29 = 64;
  if ( v29 >= (unsigned int)v4 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 1312);
    for ( i = &v5[3 * v4]; i != v5; v5 += 3 )
      *v5 = -4096;
    *(_QWORD *)(a1 + 1320) = 0;
    goto LABEL_7;
  }
  v30 = v3 - 1;
  if ( !v30 )
  {
    v31 = *(_QWORD **)(a1 + 1312);
    v32 = 64;
LABEL_47:
    sub_C7D6A0((__int64)v31, 24 * v4, 8);
    v33 = ((((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 16;
    v34 = (v33
         | (((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 1328) = v34;
    v35 = (_QWORD *)sub_C7D670(24 * v34, 8);
    v36 = *(unsigned int *)(a1 + 1328);
    *(_QWORD *)(a1 + 1320) = 0;
    *(_QWORD *)(a1 + 1312) = v35;
    for ( j = &v35[3 * v36]; j != v35; v35 += 3 )
    {
      if ( v35 )
        *v35 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v30, v30);
  v31 = *(_QWORD **)(a1 + 1312);
  v32 = 1 << (33 - (v30 ^ 0x1F));
  if ( v32 < 64 )
    v32 = 64;
  if ( (_DWORD)v4 != v32 )
    goto LABEL_47;
  *(_QWORD *)(a1 + 1320) = 0;
  v66 = &v31[3 * v4];
  do
  {
    if ( v31 )
      *v31 = -4096;
    v31 += 3;
  }
  while ( v66 != v31 );
LABEL_7:
  v7 = *(_QWORD *)(a2 + 56);
  v76 = a2 + 48;
  while ( v7 != v76 )
  {
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 8);
    if ( *(_BYTE *)(v8 - 24) != 85 )
      continue;
    v9 = *(_QWORD *)(v8 - 56);
    if ( !v9 || *(_BYTE *)v9 || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v8 + 56) || (*(_BYTE *)(v9 + 33) & 0x20) == 0 )
      continue;
    v10 = *(_DWORD *)(v9 + 36);
    if ( v7 == *(_QWORD *)(v8 + 16) + 48LL || !v7 )
      v11 = 0;
    else
      v11 = v7 - 24;
    v75 = (_QWORD *)(v8 - 24);
    sub_D5F1F0(a1 + 56, v11);
    if ( v10 != 8975 )
    {
      if ( v10 > 0x230F )
      {
        if ( v10 - 9549 > 0x12 || ((1LL << ((unsigned __int8)v10 - 77)) & 0x40011) == 0 )
          continue;
        v12 = *(_DWORD *)(v8 - 20) & 0x7FFFFFF;
        v69 = v75[4 * (1 - v12)];
        v68 = v75[4 * (2 - v12)];
        v72 = *(_QWORD *)(v8 - 32 * v12 - 24);
        v13 = sub_2A8A890(a1, (__int64)v75);
        v80 = 257;
        v14 = v13;
        v15 = sub_BD2C40(80, unk_3F10A10);
        v16 = (__int64)v15;
        if ( v15 )
          sub_B4D3C0((__int64)v15, v69, v68, 0, v14, v69, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 144) + 16LL))(
          *(_QWORD *)(a1 + 144),
          v16,
          v79,
          *(_QWORD *)(a1 + 112),
          *(_QWORD *)(a1 + 120));
        v17 = *(_QWORD *)(a1 + 56);
        v18 = v17 + 16LL * *(unsigned int *)(a1 + 64);
        if ( v17 != v18 )
        {
          v70 = v7;
          v19 = *(_QWORD *)(a1 + 56);
          do
          {
            v20 = *(_QWORD *)(v19 + 8);
            v21 = *(_DWORD *)v19;
            v19 += 16;
            sub_B99FD0(v16, v21, v20);
          }
          while ( v18 != v19 );
LABEL_27:
          v7 = v70;
          goto LABEL_28;
        }
        goto LABEL_28;
      }
      if ( v10 - 8937 > 1 )
        continue;
    }
    v38 = *(_DWORD *)(v8 - 20) & 0x7FFFFFF;
    v71 = v75[4 * (1 - v38)];
    v72 = *(_QWORD *)(v8 - 32 * v38 - 24);
    v39 = sub_2A8A890(a1, (__int64)v75);
    v78 = 257;
    v40 = *(_QWORD *)(v8 - 16);
    v41 = v39;
    v80 = 257;
    v42 = sub_BD2C40(80, 1u);
    v16 = (__int64)v42;
    if ( v42 )
      sub_B4D190((__int64)v42, v40, v71, (__int64)v79, 0, v41, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 144) + 16LL))(
      *(_QWORD *)(a1 + 144),
      v16,
      v77,
      *(_QWORD *)(a1 + 112),
      *(_QWORD *)(a1 + 120));
    v43 = *(_QWORD *)(a1 + 56);
    if ( v43 != v43 + 16LL * *(unsigned int *)(a1 + 64) )
    {
      v70 = v7;
      v44 = *(_QWORD *)(a1 + 56);
      v45 = v43 + 16LL * *(unsigned int *)(a1 + 64);
      do
      {
        v46 = *(_QWORD *)(v44 + 8);
        v47 = *(_DWORD *)v44;
        v44 += 16;
        sub_B99FD0(v16, v47, v46);
      }
      while ( v45 != v44 );
      goto LABEL_27;
    }
LABEL_28:
    v22 = *(_QWORD *)(v72 + 24);
    if ( *(_DWORD *)(v72 + 32) > 0x40u )
      v22 = **(_QWORD **)(v72 + 24);
    if ( v10 == 8975 || v10 == 9567 )
    {
      v24 = *(_DWORD *)(a1 + 1328);
      v23 = 0;
      if ( !v24 )
        goto LABEL_60;
    }
    else
    {
      if ( v10 == 8937 || v10 == 9549 )
      {
        v23 = 1;
      }
      else
      {
        if ( v10 != 8938 && v10 != 9553 )
          BUG();
        v23 = 2;
      }
      v24 = *(_DWORD *)(a1 + 1328);
      if ( !v24 )
      {
LABEL_60:
        ++*(_QWORD *)(a1 + 1304);
        goto LABEL_61;
      }
    }
    v25 = *(_QWORD *)(a1 + 1312);
    v26 = (v24 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v27 = (__int64 *)(v25 + 24LL * v26);
    v28 = *v27;
    if ( *v27 != v16 )
    {
      v57 = 1;
      v53 = 0;
      while ( v28 != -4096 )
      {
        if ( v53 || v28 != -8192 )
          v27 = v53;
        v26 = (v24 - 1) & (v57 + v26);
        v28 = *(_QWORD *)(v25 + 24LL * v26);
        if ( v28 == v16 )
          goto LABEL_39;
        ++v57;
        v53 = v27;
        v27 = (__int64 *)(v25 + 24LL * v26);
      }
      v58 = *(_DWORD *)(a1 + 1320);
      if ( !v53 )
        v53 = v27;
      ++*(_QWORD *)(a1 + 1304);
      v52 = v58 + 1;
      if ( 4 * (v58 + 1) >= 3 * v24 )
      {
LABEL_61:
        v73 = v22;
        sub_2A8ACD0(v67, 2 * v24);
        v48 = *(_DWORD *)(a1 + 1328);
        if ( !v48 )
          goto LABEL_108;
        v49 = v48 - 1;
        v50 = *(_QWORD *)(a1 + 1312);
        v22 = v73;
        v51 = v49 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v52 = *(_DWORD *)(a1 + 1320) + 1;
        v53 = (__int64 *)(v50 + 24LL * v51);
        v54 = *v53;
        if ( *v53 != v16 )
        {
          v55 = 1;
          v56 = 0;
          while ( v54 != -4096 )
          {
            if ( v54 == -8192 && !v56 )
              v56 = v53;
            v51 = v49 & (v55 + v51);
            v53 = (__int64 *)(v50 + 24LL * v51);
            v54 = *v53;
            if ( *v53 == v16 )
              goto LABEL_76;
            ++v55;
          }
          if ( v56 )
            v53 = v56;
        }
      }
      else if ( v24 - *(_DWORD *)(a1 + 1324) - v52 <= v24 >> 3 )
      {
        v74 = v22;
        sub_2A8ACD0(v67, v24);
        v59 = *(_DWORD *)(a1 + 1328);
        if ( !v59 )
        {
LABEL_108:
          ++*(_DWORD *)(a1 + 1320);
          BUG();
        }
        v60 = v59 - 1;
        v61 = *(_QWORD *)(a1 + 1312);
        v62 = 1;
        v63 = (v59 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v22 = v74;
        v52 = *(_DWORD *)(a1 + 1320) + 1;
        v64 = 0;
        v53 = (__int64 *)(v61 + 24LL * v63);
        v65 = *v53;
        if ( *v53 != v16 )
        {
          while ( v65 != -4096 )
          {
            if ( !v64 && v65 == -8192 )
              v64 = v53;
            v63 = v60 & (v62 + v63);
            v53 = (__int64 *)(v61 + 24LL * v63);
            v65 = *v53;
            if ( *v53 == v16 )
              goto LABEL_76;
            ++v62;
          }
          if ( v64 )
            v53 = v64;
        }
      }
LABEL_76:
      *(_DWORD *)(a1 + 1320) = v52;
      if ( *v53 != -4096 )
        --*(_DWORD *)(a1 + 1324);
      *v53 = v16;
      v53[1] = v22;
      *((_DWORD *)v53 + 4) = v23;
    }
LABEL_39:
    sub_BD84D0((__int64)v75, v16);
    sub_B43D60(v75);
  }
}
