// Function: sub_352B6D0
// Address: 0x352b6d0
//
__int64 __fastcall sub_352B6D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // r13
  __int64 v5; // rbx
  int v6; // r15d
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r10
  int v9; // eax
  void (__fastcall *v10)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // rdi
  int v15; // r11d
  __int64 *v16; // r10
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rdx
  int v21; // ecx
  int v22; // r9d
  int v23; // r9d
  __int64 v24; // r10
  unsigned int v25; // edx
  __int64 v26; // r8
  int v27; // edi
  __int64 *v28; // rsi
  int v29; // r8d
  int v30; // r8d
  __int64 v31; // r9
  int v32; // edx
  __int64 *v33; // rdi
  __int64 v34; // rbx
  __int64 v35; // rsi
  unsigned __int64 v37; // [rsp+10h] [rbp-B0h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  _QWORD v39[4]; // [rsp+20h] [rbp-A0h] BYREF
  char v40; // [rsp+40h] [rbp-80h]
  char v41; // [rsp+41h] [rbp-7Fh]
  _BYTE v42[16]; // [rsp+50h] [rbp-70h] BYREF
  void (__fastcall *v43)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-60h]
  _QWORD v44[2]; // [rsp+70h] [rbp-50h] BYREF
  void (__fastcall *v45)(_QWORD *, _QWORD *, __int64); // [rsp+80h] [rbp-40h]

  v2 = 0;
  v4 = *(_QWORD *)(a2 + 32);
  v38 = *(_QWORD *)(*(_QWORD *)(a1 + 144) + 32LL);
  v5 = v4 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v5 == v4 )
    return 0;
  do
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)v4 && (*(_BYTE *)(v4 + 3) & 0x10) == 0 )
      {
        v6 = *(_DWORD *)(v4 + 8);
        if ( v6 < 0 )
        {
          v7 = sub_2EBEE90(v38, v6);
          if ( v7 )
          {
            v37 = v7;
            if ( (unsigned int)sub_352B010(v7) != 3 )
              break;
          }
        }
      }
      v4 += 40;
      if ( v5 == v4 )
        goto LABEL_23;
    }
    v8 = v37;
    if ( (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x20) == 0 )
    {
      v9 = *(_DWORD *)(a2 + 44);
      if ( (v9 & 0x20000) != 0
        || ((v9 & 4) != 0 || (v9 & 8) == 0
          ? (v12 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 36) & 1LL)
          : (LOBYTE(v12) = sub_2E88A90(a2, 0x1000000000LL, 1), v8 = v37),
            !(_BYTE)v12) )
      {
        sub_2EE7340((__int64)v42, a1 + 144, v6);
        sub_2EE7320(v44, a1 + 144, a2);
        v41 = 1;
        v39[0] = "Convergence control tokens can only be used by convergent operations.";
        v40 = 3;
        sub_352B2E0((_BYTE *)a1, (__int64)v39, v42, 2);
        v10 = v45;
        if ( !v45 )
          goto LABEL_14;
        goto LABEL_13;
      }
    }
    if ( v2 )
    {
      sub_2EE7340((__int64)v42, a1 + 144, v6);
      sub_2EE7320(v44, a1 + 144, a2);
      v41 = 1;
      v39[0] = "An operation can use at most one convergence control token.";
      v40 = 3;
      sub_352B2E0((_BYTE *)a1, (__int64)v39, v42, 2);
      v10 = v45;
      if ( !v45 )
      {
LABEL_14:
        if ( v43 )
          v43(v42, v42, 3);
        return 0;
      }
LABEL_13:
      v10(v44, v44, 3);
      goto LABEL_14;
    }
    v4 += 40;
    v2 = v8;
  }
  while ( v5 != v4 );
LABEL_23:
  if ( !v2 )
    return 0;
  v13 = *(_DWORD *)(a1 + 184);
  v14 = a1 + 160;
  if ( !v13 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_45;
  }
  v15 = 1;
  v16 = 0;
  v17 = *(_QWORD *)(a1 + 168);
  v18 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = (__int64 *)(v17 + 16LL * v18);
  v20 = *v19;
  if ( a2 == *v19 )
    goto LABEL_26;
  while ( v20 != -4096 )
  {
    if ( !v16 && v20 == -8192 )
      v16 = v19;
    v18 = (v13 - 1) & (v15 + v18);
    v19 = (__int64 *)(v17 + 16LL * v18);
    v20 = *v19;
    if ( a2 == *v19 )
      goto LABEL_26;
    ++v15;
  }
  if ( v16 )
    v19 = v16;
  ++*(_QWORD *)(a1 + 160);
  v21 = *(_DWORD *)(a1 + 176) + 1;
  if ( 4 * v21 >= 3 * v13 )
  {
LABEL_45:
    sub_352B4F0(v14, 2 * v13);
    v22 = *(_DWORD *)(a1 + 184);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 168);
      v25 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = *(_DWORD *)(a1 + 176) + 1;
      v19 = (__int64 *)(v24 + 16LL * v25);
      v26 = *v19;
      if ( a2 != *v19 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( v26 == -8192 && !v28 )
            v28 = v19;
          v25 = v23 & (v27 + v25);
          v19 = (__int64 *)(v24 + 16LL * v25);
          v26 = *v19;
          if ( a2 == *v19 )
            goto LABEL_41;
          ++v27;
        }
        if ( v28 )
          v19 = v28;
      }
      goto LABEL_41;
    }
    goto LABEL_68;
  }
  if ( v13 - *(_DWORD *)(a1 + 180) - v21 <= v13 >> 3 )
  {
    sub_352B4F0(v14, v13);
    v29 = *(_DWORD *)(a1 + 184);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 168);
      v32 = 1;
      v33 = 0;
      LODWORD(v34) = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = *(_DWORD *)(a1 + 176) + 1;
      v19 = (__int64 *)(v31 + 16LL * (unsigned int)v34);
      v35 = *v19;
      if ( a2 != *v19 )
      {
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v33 )
            v33 = v19;
          v34 = v30 & (unsigned int)(v34 + v32);
          v19 = (__int64 *)(v31 + 16 * v34);
          v35 = *v19;
          if ( a2 == *v19 )
            goto LABEL_41;
          ++v32;
        }
        if ( v33 )
          v19 = v33;
      }
      goto LABEL_41;
    }
LABEL_68:
    ++*(_DWORD *)(a1 + 176);
    BUG();
  }
LABEL_41:
  *(_DWORD *)(a1 + 176) = v21;
  if ( *v19 != -4096 )
    --*(_DWORD *)(a1 + 180);
  *v19 = a2;
  v19[1] = 0;
LABEL_26:
  v19[1] = v2;
  return v2;
}
