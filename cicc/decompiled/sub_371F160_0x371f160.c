// Function: sub_371F160
// Address: 0x371f160
//
__int64 __fastcall sub_371F160(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v8; // r15
  unsigned __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 *v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // r14d
  unsigned int v23; // esi
  __int64 v24; // r8
  int v25; // r11d
  __int64 *v26; // rdx
  unsigned int v27; // edi
  __int64 *v28; // rax
  __int64 v29; // rcx
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  int v33; // eax
  int v34; // ecx
  int v35; // eax
  int v36; // esi
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // r8
  int v40; // r10d
  __int64 *v41; // r9
  int v42; // eax
  int v43; // eax
  __int64 v44; // rdi
  int v45; // r9d
  unsigned int v46; // r15d
  __int64 *v47; // r8
  __int64 v48; // rsi
  __int64 v49[4]; // [rsp+10h] [rbp-60h] BYREF
  char v50; // [rsp+30h] [rbp-40h]
  char v51; // [rsp+31h] [rbp-3Fh]

  if ( a5 )
  {
    v14 = a4;
    v51 = 1;
    v49[0] = (__int64)"pmk";
    v50 = 3;
    v12 = sub_371CDC0(0x22D9u, 0, 0, 0, 0, (__int64)v49, a4);
  }
  else
  {
    v8 = a3 + 48;
    v9 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a3 + 48 == v9 )
    {
      v11 = 0;
    }
    else
    {
      if ( !v9 )
        BUG();
      v10 = *(unsigned __int8 *)(v9 - 24);
      v11 = v9 - 24;
      if ( (unsigned int)(v10 - 30) >= 0xB )
        v11 = 0;
    }
    v51 = 1;
    v49[0] = (__int64)"pmk";
    v50 = 3;
    v12 = sub_371CDC0(0x22D9u, 0, 0, 0, 0, (__int64)v49, v11);
    v13 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 == v13 )
      goto LABEL_70;
    if ( !v13 )
      BUG();
    v14 = v13 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_70:
      BUG();
  }
  v15 = *(_QWORD *)(v14 + 48);
  v16 = (__int64 *)(v12 + 48);
  v49[0] = v15;
  if ( !v15 )
  {
    if ( v16 == v49 )
      goto LABEL_15;
    v31 = *(_QWORD *)(v12 + 48);
    if ( !v31 )
      goto LABEL_15;
LABEL_23:
    sub_B91220(v12 + 48, v31);
    goto LABEL_24;
  }
  sub_B96E90((__int64)v49, v15, 1);
  if ( v16 == v49 )
  {
    if ( v49[0] )
      sub_B91220((__int64)v49, v49[0]);
    goto LABEL_15;
  }
  v31 = *(_QWORD *)(v12 + 48);
  if ( v31 )
    goto LABEL_23;
LABEL_24:
  v32 = (unsigned __int8 *)v49[0];
  *(_QWORD *)(v12 + 48) = v49[0];
  if ( v32 )
    sub_B976B0((__int64)v49, v32, v12 + 48);
LABEL_15:
  v17 = (_QWORD *)sub_22077B0(0x10u);
  v18 = *(_QWORD *)(a2 + 40);
  v19 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)a1 = v12;
  v17[1] = v12;
  *v17 = v18;
  v20 = 0;
  *(_QWORD *)(a2 + 40) = v17;
  v21 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
  if ( (unsigned int)v21 < *(_DWORD *)(v19 + 32) )
    v20 = *(_QWORD *)(*(_QWORD *)(v19 + 24) + 8 * v21);
  v22 = *(_DWORD *)(a2 + 120);
  v23 = *(_DWORD *)(a2 + 112);
  *(_QWORD *)(a1 + 8) = v20;
  *(_DWORD *)(a2 + 120) = v22 + 1;
  if ( !v23 )
  {
    ++*(_QWORD *)(a2 + 88);
    goto LABEL_43;
  }
  v24 = *(_QWORD *)(a2 + 96);
  v25 = 1;
  v26 = 0;
  v27 = (v23 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v28 = (__int64 *)(v24 + 16LL * v27);
  v29 = *v28;
  if ( v12 == *v28 )
  {
LABEL_19:
    v22 = *((_DWORD *)v28 + 2);
    goto LABEL_20;
  }
  while ( v29 != -4096 )
  {
    if ( v29 == -8192 && !v26 )
      v26 = v28;
    v27 = (v23 - 1) & (v25 + v27);
    v28 = (__int64 *)(v24 + 16LL * v27);
    v29 = *v28;
    if ( v12 == *v28 )
      goto LABEL_19;
    ++v25;
  }
  if ( !v26 )
    v26 = v28;
  v33 = *(_DWORD *)(a2 + 104);
  ++*(_QWORD *)(a2 + 88);
  v34 = v33 + 1;
  if ( 4 * (v33 + 1) >= 3 * v23 )
  {
LABEL_43:
    sub_A41E30(a2 + 88, 2 * v23);
    v35 = *(_DWORD *)(a2 + 112);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a2 + 96);
      v38 = (v35 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v34 = *(_DWORD *)(a2 + 104) + 1;
      v26 = (__int64 *)(v37 + 16LL * v38);
      v39 = *v26;
      if ( v12 != *v26 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -4096 )
        {
          if ( !v41 && v39 == -8192 )
            v41 = v26;
          v38 = v36 & (v40 + v38);
          v26 = (__int64 *)(v37 + 16LL * v38);
          v39 = *v26;
          if ( v12 == *v26 )
            goto LABEL_38;
          ++v40;
        }
        if ( v41 )
          v26 = v41;
      }
      goto LABEL_38;
    }
    goto LABEL_67;
  }
  if ( v23 - *(_DWORD *)(a2 + 108) - v34 <= v23 >> 3 )
  {
    sub_A41E30(a2 + 88, v23);
    v42 = *(_DWORD *)(a2 + 112);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a2 + 96);
      v45 = 1;
      v46 = v43 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v47 = 0;
      v34 = *(_DWORD *)(a2 + 104) + 1;
      v26 = (__int64 *)(v44 + 16LL * v46);
      v48 = *v26;
      if ( v12 != *v26 )
      {
        while ( v48 != -4096 )
        {
          if ( !v47 && v48 == -8192 )
            v47 = v26;
          v46 = v43 & (v45 + v46);
          v26 = (__int64 *)(v44 + 16LL * v46);
          v48 = *v26;
          if ( v12 == *v26 )
            goto LABEL_38;
          ++v45;
        }
        if ( v47 )
          v26 = v47;
      }
      goto LABEL_38;
    }
LABEL_67:
    ++*(_DWORD *)(a2 + 104);
    BUG();
  }
LABEL_38:
  *(_DWORD *)(a2 + 104) = v34;
  if ( *v26 != -4096 )
    --*(_DWORD *)(a2 + 108);
  *v26 = v12;
  *((_DWORD *)v26 + 2) = v22;
LABEL_20:
  *(_DWORD *)(a1 + 16) = v22;
  return a1;
}
