// Function: sub_1C74800
// Address: 0x1c74800
//
__int64 __fastcall sub_1C74800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r12
  __int64 v16; // r13
  int v17; // eax
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rdx
  int v29; // eax
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v40; // [rsp+10h] [rbp-60h]
  const char *v42; // [rsp+20h] [rbp-50h] BYREF
  char v43; // [rsp+30h] [rbp-40h]
  char v44; // [rsp+31h] [rbp-3Fh]

  v9 = *(_QWORD *)(a1 + 48);
  v44 = 1;
  if ( v9 )
    v9 -= 24;
  v43 = 3;
  v42 = "splitPhi";
  v40 = v9;
  v10 = sub_1648B60(64);
  v14 = v40;
  v15 = v10;
  if ( v10 )
  {
    v16 = v10;
    sub_15F1EA0(v10, a2, 53, 0, 0, v40);
    *(_DWORD *)(v15 + 56) = 2;
    sub_164B780(v15, (__int64 *)&v42);
    a2 = *(unsigned int *)(v15 + 56);
    sub_1648880(v15, a2, 1);
  }
  else
  {
    v16 = 0;
  }
  v17 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  if ( v17 == *(_DWORD *)(v15 + 56) )
  {
    sub_15F55D0(v15, a2, v11, v12, v13, v14);
    v17 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  }
  v18 = (v17 + 1) & 0xFFFFFFF;
  v19 = v18 | *(_DWORD *)(v15 + 20) & 0xF0000000;
  *(_DWORD *)(v15 + 20) = v19;
  if ( (v19 & 0x40000000) != 0 )
    v20 = *(_QWORD *)(v15 - 8);
  else
    v20 = v16 - 24 * v18;
  v21 = (_QWORD *)(v20 + 24LL * (unsigned int)(v18 - 1));
  if ( *v21 )
  {
    v22 = v21[1];
    v23 = v21[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v23 = v22;
    if ( v22 )
    {
      a2 = *(_QWORD *)(v22 + 16) & 3LL;
      *(_QWORD *)(v22 + 16) = a2 | v23;
    }
  }
  *v21 = a3;
  if ( a3 )
  {
    v24 = *(_QWORD *)(a3 + 8);
    a2 = a3 + 8;
    v21[1] = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = (unsigned __int64)(v21 + 1) | *(_QWORD *)(v24 + 16) & 3LL;
    v21[2] = a2 | v21[2] & 3LL;
    *(_QWORD *)(a3 + 8) = v21;
  }
  v25 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  v26 = (unsigned int)(v25 - 1);
  if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
    v27 = *(_QWORD *)(v15 - 8);
  else
    v27 = v16 - 24 * v25;
  v28 = 3LL * *(unsigned int *)(v15 + 56);
  *(_QWORD *)(v27 + 8 * v26 + 24LL * *(unsigned int *)(v15 + 56) + 8) = a4;
  v29 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  if ( v29 == *(_DWORD *)(v15 + 56) )
  {
    sub_15F55D0(v15, a2, v28, v27, v13, v14);
    v29 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  }
  v30 = (v29 + 1) & 0xFFFFFFF;
  v31 = v30 | *(_DWORD *)(v15 + 20) & 0xF0000000;
  *(_DWORD *)(v15 + 20) = v31;
  if ( (v31 & 0x40000000) != 0 )
    v32 = *(_QWORD *)(v15 - 8);
  else
    v32 = v16 - 24 * v30;
  v33 = (_QWORD *)(v32 + 24LL * (unsigned int)(v30 - 1));
  if ( *v33 )
  {
    v34 = v33[1];
    v35 = v33[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v35 = v34;
    if ( v34 )
      *(_QWORD *)(v34 + 16) = *(_QWORD *)(v34 + 16) & 3LL | v35;
  }
  *v33 = a5;
  if ( a5 )
  {
    v36 = *(_QWORD *)(a5 + 8);
    v33[1] = v36;
    if ( v36 )
      *(_QWORD *)(v36 + 16) = (unsigned __int64)(v33 + 1) | *(_QWORD *)(v36 + 16) & 3LL;
    v33[2] = (a5 + 8) | v33[2] & 3LL;
    *(_QWORD *)(a5 + 8) = v33;
  }
  v37 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
    v38 = *(_QWORD *)(v15 - 8);
  else
    v38 = v16 - 24 * v37;
  *(_QWORD *)(v38 + 8LL * (unsigned int)(v37 - 1) + 24LL * *(unsigned int *)(v15 + 56) + 8) = a6;
  return v15;
}
