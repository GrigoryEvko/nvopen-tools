// Function: sub_A810C0
// Address: 0xa810c0
//
__int64 __fastcall sub_A810C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v7; // eax
  unsigned int v8; // ecx
  __int64 result; // rax
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  char v21; // r8
  __int64 v22; // r15
  unsigned int *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rsi
  int v27; // r13d
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r13
  __int64 v31; // rdi
  unsigned int v32; // edx
  bool v33; // al
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v39; // [rsp+Eh] [rbp-A2h]
  char v40; // [rsp+Fh] [rbp-A1h]
  __int64 v41; // [rsp+10h] [rbp-A0h]
  __int64 v42; // [rsp+18h] [rbp-98h]
  int v43; // [rsp+20h] [rbp-90h]
  int v44; // [rsp+24h] [rbp-8Ch]
  __int64 v46; // [rsp+30h] [rbp-80h] BYREF
  __int64 v47; // [rsp+38h] [rbp-78h]
  __int64 v48; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+48h] [rbp-68h]
  unsigned __int64 v50; // [rsp+50h] [rbp-60h] BYREF
  __int64 v51; // [rsp+58h] [rbp-58h]
  __int16 v52; // [rsp+70h] [rbp-40h]

  if ( a2 <= 6 )
    goto LABEL_21;
  if ( *(_DWORD *)a1 == 1714320228 && *(_WORD *)(a1 + 4) == 25697 && *(_BYTE *)(a1 + 6) == 100 )
  {
LABEL_66:
    v44 = 11;
    goto LABEL_22;
  }
  if ( *(_DWORD *)a1 == 1714320228 && *(_WORD *)(a1 + 4) == 26989 && *(_BYTE *)(a1 + 6) == 110 )
    goto LABEL_68;
  if ( *(_DWORD *)a1 == 1714320228 && *(_WORD *)(a1 + 4) == 24941 && *(_BYTE *)(a1 + 6) == 120 )
    goto LABEL_84;
  if ( a2 <= 0xA )
    goto LABEL_21;
  if ( *(_QWORD *)a1 == 0x692E63696D6F7461LL && *(_WORD *)(a1 + 8) == 25454 && *(_BYTE *)(a1 + 10) == 46 )
  {
    v44 = 15;
    goto LABEL_22;
  }
  if ( *(_QWORD *)a1 == 0x642E63696D6F7461LL && *(_WORD *)(a1 + 8) == 25445 && *(_BYTE *)(a1 + 10) == 46 )
  {
    v44 = 16;
    goto LABEL_22;
  }
  if ( a2 <= 0x11 )
  {
LABEL_21:
    if ( a2 <= 0xF )
      goto LABEL_22;
    v34 = *(_QWORD *)a1 ^ 0x6F74612E74616C66LL;
    if ( !(v34 | *(_QWORD *)(a1 + 8) ^ 0x646461662E63696DLL) )
      goto LABEL_66;
    if ( v34 | *(_QWORD *)(a1 + 8) ^ 0x6E696D662E63696DLL )
    {
LABEL_16:
      v7 = 13;
      if ( *(_QWORD *)a1 ^ 0x6F74612E74616C66LL | *(_QWORD *)(a1 + 8) ^ 0x78616D662E63696DLL )
        v7 = 0;
      v8 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
      v44 = v7;
      if ( v8 > 2 )
        goto LABEL_23;
      return 0;
    }
LABEL_68:
    v44 = 14;
    goto LABEL_22;
  }
  if ( !(*(_QWORD *)a1 ^ 0x612E6C61626F6C67LL | *(_QWORD *)(a1 + 8) ^ 0x61662E63696D6F74LL)
    && *(_WORD *)(a1 + 16) == 25700
    || !(*(_QWORD *)a1 ^ 0x6F74612E74616C66LL | *(_QWORD *)(a1 + 8) ^ 0x646461662E63696DLL) )
  {
    goto LABEL_66;
  }
  if ( !(*(_QWORD *)a1 ^ 0x612E6C61626F6C67LL | *(_QWORD *)(a1 + 8) ^ 0x6D662E63696D6F74LL)
    && *(_WORD *)(a1 + 16) == 28265
    || !(*(_QWORD *)a1 ^ 0x6F74612E74616C66LL | *(_QWORD *)(a1 + 8) ^ 0x6E696D662E63696DLL) )
  {
    goto LABEL_68;
  }
  if ( *(_QWORD *)a1 ^ 0x612E6C61626F6C67LL | *(_QWORD *)(a1 + 8) ^ 0x6D662E63696D6F74LL || *(_WORD *)(a1 + 16) != 30817 )
    goto LABEL_16;
LABEL_84:
  v44 = 13;
LABEL_22:
  v8 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  if ( v8 <= 2 )
    return 0;
LABEL_23:
  v10 = *(_QWORD *)(a3 - 32LL * v8);
  v42 = *(_QWORD *)(v10 + 8);
  result = 0;
  if ( *(_BYTE *)(v42 + 8) != 14 )
    return result;
  v11 = *(_QWORD *)(a3 + 32 * (1LL - v8));
  if ( *(_QWORD *)(v11 + 8) != *(_QWORD *)(a3 + 8) )
    return result;
  if ( v8 == 3 )
    goto LABEL_69;
  v12 = *(_QWORD *)(a3 + 32 * (2LL - v8));
  if ( *(_BYTE *)v12 == 17 )
  {
    if ( v8 <= 5 )
    {
      v40 = 0;
LABEL_29:
      v13 = *(_QWORD **)(v12 + 24);
      if ( *(_DWORD *)(v12 + 32) > 0x40u )
        v13 = (_QWORD *)*v13;
      v43 = 7;
      if ( v13 != (_QWORD *)3 )
      {
        if ( (unsigned __int64)v13 - 2 > 5 )
          LODWORD(v13) = 7;
        v43 = (int)v13;
      }
      goto LABEL_35;
    }
    v31 = *(_QWORD *)(a3 + 32 * (4LL - v8));
    if ( *(_BYTE *)v31 != 17 )
    {
      v40 = 1;
      goto LABEL_29;
    }
    goto LABEL_60;
  }
  if ( v8 <= 5 )
  {
LABEL_69:
    v40 = 0;
    v43 = 7;
    goto LABEL_35;
  }
  v31 = *(_QWORD *)(a3 + 32 * (4LL - v8));
  if ( *(_BYTE *)v31 == 17 )
  {
    v12 = 0;
LABEL_60:
    v32 = *(_DWORD *)(v31 + 32);
    if ( v32 <= 0x40 )
      v33 = *(_QWORD *)(v31 + 24) == 0;
    else
      v33 = v32 == (unsigned int)sub_C444A0(v31 + 24);
    v40 = !v33;
    if ( v12 )
      goto LABEL_29;
    goto LABEL_91;
  }
  v40 = 1;
LABEL_91:
  v43 = 7;
LABEL_35:
  v14 = sub_B2BE50(a4);
  v15 = *(_QWORD *)(a3 + 8);
  v16 = v14;
  v41 = v15;
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 && (unsigned __int8)sub_BCAC40(*(_QWORD *)(v15 + 24), 16) )
  {
    v35 = *(_DWORD *)(v41 + 32);
    BYTE4(v47) = *(_BYTE *)(v41 + 8) == 18;
    LODWORD(v47) = v35;
    v36 = sub_BCB150(v16);
    v37 = sub_BCE1B0(v36, v47);
    v52 = 257;
    v11 = sub_A7EAA0((unsigned int **)a5, 0x31u, v11, v37, (__int64)&v50, 0, v48, 0);
  }
  v38 = (unsigned __int8)sub_B6F810(v16, "agent", 5);
  v17 = sub_AA4E30(*(_QWORD *)(a5 + 48));
  v18 = sub_9208B0(v17, *(_QWORD *)(v11 + 8));
  v51 = v19;
  v50 = (unsigned __int64)(v18 + 7) >> 3;
  v20 = sub_CA1930(&v50);
  v21 = -1;
  if ( v20 )
  {
    _BitScanReverse64(&v20, v20);
    v21 = 63 - (v20 ^ 0x3F);
  }
  v39 = v21;
  v52 = 257;
  v22 = sub_BD2C40(80, unk_3F148C0);
  if ( v22 )
    sub_B4D750(v22, v44, v10, v11, v39, v43, v38, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
    *(_QWORD *)(a5 + 88),
    v22,
    &v50,
    *(_QWORD *)(a5 + 56),
    *(_QWORD *)(a5 + 64));
  v23 = *(unsigned int **)a5;
  v24 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
  if ( *(_QWORD *)a5 != v24 )
  {
    do
    {
      v25 = *((_QWORD *)v23 + 1);
      v26 = *v23;
      v23 += 4;
      sub_B99FD0(v22, v26, v25);
    }
    while ( (unsigned int *)v24 != v23 );
  }
  v27 = *(_DWORD *)(v42 + 8) >> 8;
  if ( v27 != 3 )
  {
    v28 = sub_B2BE50(a4);
    v29 = sub_B9C770(v28, 0, 0, 0, 1);
    sub_B9A090(v22, "amdgpu.no.fine.grained.memory", 29, v29);
    if ( v44 == 11 && *(_BYTE *)(v41 + 8) == 2 )
      sub_B9A090(v22, "amdgpu.ignore.denormal.mode", 27, v29);
    if ( !v27 )
    {
      v46 = sub_B2BE50(a4);
      LODWORD(v51) = 32;
      v50 = 6;
      v49 = 32;
      v48 = 5;
      v30 = sub_B8C820(&v46, &v48, &v50);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      if ( (unsigned int)v51 > 0x40 && v50 )
        j_j___libc_free_0_0(v50);
      sub_B99FD0(v22, 41, v30);
    }
  }
  if ( v40 )
    *(_WORD *)(v22 + 2) |= 1u;
  v52 = 257;
  return sub_A7EAA0((unsigned int **)a5, 0x31u, v22, v41, (__int64)&v50, 0, v48, 0);
}
