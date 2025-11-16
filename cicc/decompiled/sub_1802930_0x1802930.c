// Function: sub_1802930
// Address: 0x1802930
//
__int64 __fastcall sub_1802930(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // r14
  int v22; // eax
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rdx
  _QWORD *v26; // rax
  __int64 v27; // rcx
  unsigned __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r12
  int v35; // eax
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // rcx
  unsigned __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // r13
  __int64 *v46; // [rsp+8h] [rbp-98h]
  __int64 v49; // [rsp+28h] [rbp-78h] BYREF
  __int64 v50; // [rsp+30h] [rbp-70h] BYREF
  __int16 v51; // [rsp+40h] [rbp-60h]
  __int64 v52; // [rsp+50h] [rbp-50h] BYREF
  __int16 v53; // [rsp+60h] [rbp-40h]

  v51 = 257;
  v53 = 257;
  v9 = sub_1648B60(64);
  v10 = v9;
  if ( v9 )
  {
    v11 = v9;
    sub_15F1EA0(v9, a1, 53, 0, 0, 0);
    *(_DWORD *)(v10 + 56) = 2;
    sub_164B780(v10, &v52);
    sub_1648880(v10, *(_DWORD *)(v10 + 56), 1);
  }
  else
  {
    v11 = 0;
  }
  v12 = a2[1];
  if ( v12 )
  {
    v46 = (__int64 *)a2[2];
    sub_157E9D0(v12 + 40, v10);
    v13 = *v46;
    v14 = *(_QWORD *)(v10 + 24) & 7LL;
    *(_QWORD *)(v10 + 32) = v46;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v10 + 24) = v13 | v14;
    *(_QWORD *)(v13 + 8) = v10 + 24;
    *v46 = *v46 & 7 | (v10 + 24);
  }
  sub_164B780(v11, &v50);
  v19 = *a2;
  if ( *a2 )
  {
    v49 = *a2;
    sub_1623A60((__int64)&v49, v19, 2);
    v20 = *(_QWORD *)(v10 + 48);
    v15 = v10 + 48;
    if ( v20 )
    {
      sub_161E7C0(v10 + 48, v20);
      v15 = v10 + 48;
    }
    v19 = v49;
    *(_QWORD *)(v10 + 48) = v49;
    if ( v19 )
      sub_1623210((__int64)&v49, (unsigned __int8 *)v19, v15);
  }
  v21 = *(_QWORD *)(a3 + 40);
  v22 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  if ( v22 == *(_DWORD *)(v10 + 56) )
  {
    sub_15F55D0(v10, v19, v15, v16, v17, v18);
    v22 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  }
  v23 = (v22 + 1) & 0xFFFFFFF;
  v24 = v23 | *(_DWORD *)(v10 + 20) & 0xF0000000;
  *(_DWORD *)(v10 + 20) = v24;
  if ( (v24 & 0x40000000) != 0 )
    v25 = *(_QWORD *)(v10 - 8);
  else
    v25 = v11 - 24 * v23;
  v26 = (_QWORD *)(v25 + 24LL * (unsigned int)(v23 - 1));
  if ( *v26 )
  {
    v27 = v26[1];
    v28 = v26[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v28 = v27;
    if ( v27 )
    {
      v19 = *(_QWORD *)(v27 + 16) & 3LL;
      *(_QWORD *)(v27 + 16) = v19 | v28;
    }
  }
  *v26 = a6;
  if ( a6 )
  {
    v29 = *(_QWORD *)(a6 + 8);
    v19 = a6 + 8;
    v26[1] = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 16) = (unsigned __int64)(v26 + 1) | *(_QWORD *)(v29 + 16) & 3LL;
    v26[2] = v19 | v26[2] & 3LL;
    *(_QWORD *)(a6 + 8) = v26;
  }
  v30 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  v31 = (unsigned int)(v30 - 1);
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v32 = *(_QWORD *)(v10 - 8);
  else
    v32 = v11 - 24 * v30;
  v33 = 3LL * *(unsigned int *)(v10 + 56);
  *(_QWORD *)(v32 + 8 * v31 + 24LL * *(unsigned int *)(v10 + 56) + 8) = v21;
  v34 = *(_QWORD *)(a5 + 40);
  v35 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  if ( v35 == *(_DWORD *)(v10 + 56) )
  {
    sub_15F55D0(v10, v19, v33, v32, v17, v18);
    v35 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  }
  v36 = (v35 + 1) & 0xFFFFFFF;
  v37 = v36 | *(_DWORD *)(v10 + 20) & 0xF0000000;
  *(_DWORD *)(v10 + 20) = v37;
  if ( (v37 & 0x40000000) != 0 )
    v38 = *(_QWORD *)(v10 - 8);
  else
    v38 = v11 - 24 * v36;
  v39 = (_QWORD *)(v38 + 24LL * (unsigned int)(v36 - 1));
  if ( *v39 )
  {
    v40 = v39[1];
    v41 = v39[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v41 = v40;
    if ( v40 )
      *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
  }
  *v39 = a4;
  if ( a4 )
  {
    v42 = *(_QWORD *)(a4 + 8);
    v39[1] = v42;
    if ( v42 )
      *(_QWORD *)(v42 + 16) = (unsigned __int64)(v39 + 1) | *(_QWORD *)(v42 + 16) & 3LL;
    v39[2] = (a4 + 8) | v39[2] & 3LL;
    *(_QWORD *)(a4 + 8) = v39;
  }
  v43 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v44 = *(_QWORD *)(v10 - 8);
  else
    v44 = v11 - 24 * v43;
  *(_QWORD *)(v44 + 8LL * (unsigned int)(v43 - 1) + 24LL * *(unsigned int *)(v10 + 56) + 8) = v34;
  return v10;
}
