// Function: sub_1CF2B90
// Address: 0x1cf2b90
//
__int64 __fastcall sub_1CF2B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r8
  int v21; // r14d
  unsigned int v22; // esi
  __int64 v23; // r8
  int v24; // r11d
  __int64 *v25; // rdx
  unsigned int v26; // edi
  __int64 *v27; // rax
  __int64 v28; // rcx
  __int64 v30; // rsi
  unsigned __int8 *v31; // rsi
  int v32; // eax
  int v33; // ecx
  int v34; // edx
  int v35; // r9d
  int v36; // eax
  int v37; // esi
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // r8
  int v41; // r10d
  __int64 *v42; // r9
  int v43; // eax
  int v44; // eax
  __int64 v45; // rdi
  int v46; // r9d
  unsigned int v47; // r15d
  __int64 *v48; // r8
  __int64 v49; // rsi
  __int64 v51[2]; // [rsp+10h] [rbp-50h] BYREF
  char v52; // [rsp+20h] [rbp-40h]
  char v53; // [rsp+21h] [rbp-3Fh]

  if ( *(_BYTE *)(a4 + 8) )
    v7 = *(_QWORD *)a4;
  else
    v7 = sub_157EBA0(a3);
  v51[0] = (__int64)"pmk";
  v53 = 1;
  v52 = 3;
  v8 = sub_1CF0110(4044, 0, 0, 0, 0, (__int64)v51, v7);
  if ( *(_BYTE *)(a4 + 8) )
    v9 = *(_QWORD *)a4;
  else
    v9 = sub_157EBA0(a3);
  v10 = *(_QWORD *)(v9 + 48);
  v11 = (__int64 *)(v8 + 48);
  v51[0] = v10;
  if ( !v10 )
  {
    if ( v11 == v51 )
      goto LABEL_9;
    v30 = *(_QWORD *)(v8 + 48);
    if ( !v30 )
      goto LABEL_9;
LABEL_20:
    sub_161E7C0(v8 + 48, v30);
    goto LABEL_21;
  }
  sub_1623A60((__int64)v51, v10, 2);
  if ( v11 == v51 )
  {
    if ( v51[0] )
      sub_161E7C0((__int64)v51, v51[0]);
    goto LABEL_9;
  }
  v30 = *(_QWORD *)(v8 + 48);
  if ( v30 )
    goto LABEL_20;
LABEL_21:
  v31 = (unsigned __int8 *)v51[0];
  *(_QWORD *)(v8 + 48) = v51[0];
  if ( v31 )
    sub_1623210((__int64)v51, v31, v8 + 48);
LABEL_9:
  v12 = (_QWORD *)sub_22077B0(16);
  v13 = *(_QWORD *)(a2 + 40);
  v14 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)a1 = v8;
  v12[1] = v8;
  *v12 = v13;
  v15 = 0;
  *(_QWORD *)(a2 + 40) = v12;
  v16 = *(unsigned int *)(v14 + 48);
  if ( !(_DWORD)v16 )
    goto LABEL_13;
  v17 = *(_QWORD *)(v14 + 32);
  v18 = (v16 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = (__int64 *)(v17 + 16LL * v18);
  v20 = *v19;
  if ( a3 == *v19 )
  {
LABEL_11:
    if ( v19 != (__int64 *)(v17 + 16 * v16) )
    {
      v15 = v19[1];
      goto LABEL_13;
    }
  }
  else
  {
    v34 = 1;
    while ( v20 != -8 )
    {
      v35 = v34 + 1;
      v18 = (v16 - 1) & (v34 + v18);
      v19 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( a3 == *v19 )
        goto LABEL_11;
      v34 = v35;
    }
  }
  v15 = 0;
LABEL_13:
  v21 = *(_DWORD *)(a2 + 120);
  v22 = *(_DWORD *)(a2 + 112);
  *(_QWORD *)(a1 + 8) = v15;
  *(_DWORD *)(a2 + 120) = v21 + 1;
  if ( !v22 )
  {
    ++*(_QWORD *)(a2 + 88);
    goto LABEL_45;
  }
  v23 = *(_QWORD *)(a2 + 96);
  v24 = 1;
  v25 = 0;
  v26 = (v22 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v27 = (__int64 *)(v23 + 16LL * v26);
  v28 = *v27;
  if ( v8 == *v27 )
  {
LABEL_15:
    v21 = *((_DWORD *)v27 + 2);
    goto LABEL_16;
  }
  while ( v28 != -8 )
  {
    if ( !v25 && v28 == -16 )
      v25 = v27;
    v26 = (v22 - 1) & (v24 + v26);
    v27 = (__int64 *)(v23 + 16LL * v26);
    v28 = *v27;
    if ( v8 == *v27 )
      goto LABEL_15;
    ++v24;
  }
  if ( !v25 )
    v25 = v27;
  v32 = *(_DWORD *)(a2 + 104);
  ++*(_QWORD *)(a2 + 88);
  v33 = v32 + 1;
  if ( 4 * (v32 + 1) >= 3 * v22 )
  {
LABEL_45:
    sub_1541C50(a2 + 88, 2 * v22);
    v36 = *(_DWORD *)(a2 + 112);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a2 + 96);
      v39 = (v36 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v33 = *(_DWORD *)(a2 + 104) + 1;
      v25 = (__int64 *)(v38 + 16LL * v39);
      v40 = *v25;
      if ( v8 != *v25 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != -8 )
        {
          if ( v40 == -16 && !v42 )
            v42 = v25;
          v39 = v37 & (v41 + v39);
          v25 = (__int64 *)(v38 + 16LL * v39);
          v40 = *v25;
          if ( v8 == *v25 )
            goto LABEL_36;
          ++v41;
        }
        if ( v42 )
          v25 = v42;
      }
      goto LABEL_36;
    }
    goto LABEL_68;
  }
  if ( v22 - *(_DWORD *)(a2 + 108) - v33 <= v22 >> 3 )
  {
    sub_1541C50(a2 + 88, v22);
    v43 = *(_DWORD *)(a2 + 112);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a2 + 96);
      v46 = 1;
      v47 = v44 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v48 = 0;
      v33 = *(_DWORD *)(a2 + 104) + 1;
      v25 = (__int64 *)(v45 + 16LL * v47);
      v49 = *v25;
      if ( v8 != *v25 )
      {
        while ( v49 != -8 )
        {
          if ( v49 == -16 && !v48 )
            v48 = v25;
          v47 = v44 & (v46 + v47);
          v25 = (__int64 *)(v45 + 16LL * v47);
          v49 = *v25;
          if ( v8 == *v25 )
            goto LABEL_36;
          ++v46;
        }
        if ( v48 )
          v25 = v48;
      }
      goto LABEL_36;
    }
LABEL_68:
    ++*(_DWORD *)(a2 + 104);
    BUG();
  }
LABEL_36:
  *(_DWORD *)(a2 + 104) = v33;
  if ( *v25 != -8 )
    --*(_DWORD *)(a2 + 108);
  *v25 = v8;
  *((_DWORD *)v25 + 2) = v21;
LABEL_16:
  *(_DWORD *)(a1 + 16) = v21;
  return a1;
}
