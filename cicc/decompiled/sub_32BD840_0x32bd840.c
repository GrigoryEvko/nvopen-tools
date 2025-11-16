// Function: sub_32BD840
// Address: 0x32bd840
//
__int64 __fastcall sub_32BD840(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  int v7; // edx
  int v9; // ecx
  unsigned __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // r14
  _QWORD *v21; // rax
  __int64 v22; // r10
  __int64 v23; // r8
  _QWORD *v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // [rsp+8h] [rbp-A8h]
  __int64 v33; // [rsp+10h] [rbp-A0h]
  __int64 v34; // [rsp+10h] [rbp-A0h]
  _QWORD *v35; // [rsp+10h] [rbp-A0h]
  _QWORD *v36; // [rsp+10h] [rbp-A0h]
  __int64 v37; // [rsp+18h] [rbp-98h]
  __int64 v38; // [rsp+18h] [rbp-98h]
  __int64 v39; // [rsp+18h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+18h] [rbp-98h]
  unsigned int v42; // [rsp+20h] [rbp-90h]
  int v43; // [rsp+20h] [rbp-90h]
  __int64 v44; // [rsp+28h] [rbp-88h]
  int v45; // [rsp+28h] [rbp-88h]
  int v46; // [rsp+38h] [rbp-78h] BYREF
  int v47; // [rsp+3Ch] [rbp-74h] BYREF
  unsigned __int16 v48[4]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v49; // [rsp+48h] [rbp-68h]
  __int64 v50; // [rsp+50h] [rbp-60h]
  __int64 v51; // [rsp+58h] [rbp-58h]
  __int64 (__fastcall **v52)(); // [rsp+60h] [rbp-50h] BYREF
  __int64 v53; // [rsp+68h] [rbp-48h]
  __int64 v54; // [rsp+70h] [rbp-40h]
  _QWORD *v55; // [rsp+78h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_QWORD *)(v2 + 40);
  if ( *(_DWORD *)(a2 + 24) != 299 )
    return 0;
  if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
    return 0;
  if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    return 0;
  v5 = *(_QWORD *)(v2 + 40);
  if ( *(_DWORD *)(v3 + 24) != 298 )
    return 0;
  if ( (*(_BYTE *)(v3 + 33) & 0xC) != 0 )
    return 0;
  if ( (*(_WORD *)(v3 + 32) & 0x380) != 0 )
    return 0;
  v6 = *(_QWORD *)(v3 + 56);
  if ( !v6 )
    return 0;
  v7 = *(_DWORD *)(v2 + 48);
  v9 = 1;
  do
  {
    if ( *(_DWORD *)(v6 + 8) == v7 )
    {
      if ( !v9 )
        return 0;
      v6 = *(_QWORD *)(v6 + 32);
      if ( !v6 )
        goto LABEL_17;
      if ( v7 == *(_DWORD *)(v6 + 8) )
        return 0;
      v9 = 0;
    }
    v6 = *(_QWORD *)(v6 + 32);
  }
  while ( v6 );
  if ( v9 == 1 )
    return 0;
LABEL_17:
  v10 = *(_WORD *)(v3 + 96);
  v11 = *(_QWORD *)(v3 + 104);
  v48[0] = v10;
  v49 = v11;
  if ( !v10
    || (unsigned __int16)(v10 - 10) > 6u
    && (unsigned __int16)(v10 - 126) > 0x31u
    && (unsigned __int16)(v10 - 208) > 0x14u )
  {
    return 0;
  }
  if ( v10 != *(_WORD *)(a2 + 96) )
    return 0;
  if ( (*(_BYTE *)(v3 + 32) & 0x10) != 0 )
    return 0;
  if ( (*(_BYTE *)(a2 + 32) & 0x10) != 0 )
    return 0;
  if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(v3 + 112)) )
    return 0;
  if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(a2 + 112)) )
    return 0;
  v12 = sub_2D5B750(v48);
  v51 = v13;
  v50 = v12;
  if ( (_BYTE)v13 )
    return 0;
  v14 = *a1;
  v46 = 0;
  v47 = 0;
  v42 = sub_327FC40(*(_QWORD **)(v14 + 64), v50);
  v44 = v15;
  v37 = a1[1];
  if ( !sub_328D6E0(v37, 0x12Au, v42) )
    return 0;
  if ( !sub_328D6E0(v37, 0x12Bu, v42) )
    return 0;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)v37 + 2200LL))(
          v37,
          298,
          *(unsigned int *)v48,
          v49) )
    return 0;
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(*(_QWORD *)a1[1] + 2200LL))(
          a1[1],
          299,
          *(unsigned int *)v48,
          v49) )
    return 0;
  v33 = a1[1];
  v38 = *(_QWORD *)(v3 + 112);
  v16 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
  if ( !(unsigned __int8)sub_2FEBB30(v33, *(_QWORD *)(*a1 + 64LL), v16, v42, v44, v38, &v46) )
    return 0;
  v34 = a1[1];
  v39 = *(_QWORD *)(a2 + 112);
  v17 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
  if ( !(unsigned __int8)sub_2FEBB30(v34, *(_QWORD *)(*a1 + 64LL), v17, v42, v44, v39, &v47) || !v46 || !v47 )
    return 0;
  v32 = *a1;
  v40 = *(_QWORD *)(v3 + 112);
  v35 = *(_QWORD **)(v3 + 40);
  sub_3285E70((__int64)&v52, v3);
  v18 = sub_33F1A60(v32, v42, v44, (unsigned int)&v52, *v35, v35[1], v35[5], v35[6], v40);
  v45 = v19;
  v20 = v18;
  sub_9C6650(&v52);
  v21 = *(_QWORD **)(a2 + 40);
  v22 = *a1;
  v23 = *(_QWORD *)(a2 + 112);
  v52 = *(__int64 (__fastcall ***)())(a2 + 80);
  v24 = v21;
  if ( v52 )
  {
    v36 = v21;
    v41 = v23;
    v43 = v22;
    sub_325F5D0((__int64 *)&v52);
    v21 = *(_QWORD **)(a2 + 40);
    v24 = v36;
    v23 = v41;
    LODWORD(v22) = v43;
  }
  LODWORD(v53) = *(_DWORD *)(a2 + 72);
  v25 = sub_33F3F90(v22, *v21, v21[1], (unsigned int)&v52, v20, v45, v24[10], v24[11], v23);
  sub_9C6650(&v52);
  sub_32B3E80((__int64)a1, v20, 1, 0, v26, v27);
  sub_32B3E80((__int64)a1, v25, 1, 0, v28, v29);
  v30 = *(_QWORD *)(*a1 + 768LL);
  v54 = *a1;
  v53 = v30;
  *(_QWORD *)(v54 + 768) = &v52;
  v31 = *a1;
  v52 = off_4A360B8;
  v55 = a1;
  sub_34161C0(v31, v5, 1, v20, 1);
  *(_QWORD *)(v54 + 768) = v53;
  return v25;
}
