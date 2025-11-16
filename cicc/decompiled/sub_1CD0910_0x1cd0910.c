// Function: sub_1CD0910
// Address: 0x1cd0910
//
__int64 __fastcall sub_1CD0910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r13
  int v19; // eax
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rcx
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32[2]; // [rsp+10h] [rbp-50h] BYREF
  char v33; // [rsp+20h] [rbp-40h]
  char v34; // [rsp+21h] [rbp-3Fh]

  v34 = 1;
  v31 = sub_15A0680(a2, 0, 1u);
  v32[0] = (__int64)"phiNode";
  v33 = 3;
  v8 = sub_1648B60(64);
  v11 = a1;
  v12 = v8;
  if ( v8 )
  {
    v30 = v8;
    sub_15F1EA0(v8, a2, 53, 0, 0, a1);
    *(_DWORD *)(v12 + 56) = 2;
    sub_164B780(v12, v32);
    sub_1648880(v12, *(_DWORD *)(v12 + 56), 1);
  }
  else
  {
    v30 = 0;
  }
  sub_1704F80(v12, v31, a4, v9, v10, v11);
  v13 = sub_157EBA0(a5);
  v34 = 1;
  v32[0] = (__int64)"baseIV";
  v33 = 3;
  v18 = sub_15FB440(11, (__int64 *)v12, a3, (__int64)v32, v13);
  v19 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  if ( v19 == *(_DWORD *)(v12 + 56) )
  {
    sub_15F55D0(v12, v12, v14, v15, v16, v17);
    v19 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  }
  v20 = (v19 + 1) & 0xFFFFFFF;
  v21 = v20 | *(_DWORD *)(v12 + 20) & 0xF0000000;
  *(_DWORD *)(v12 + 20) = v21;
  if ( (v21 & 0x40000000) != 0 )
    v22 = *(_QWORD *)(v12 - 8);
  else
    v22 = v30 - 24 * v20;
  v23 = (__int64 *)(v22 + 24LL * (unsigned int)(v20 - 1));
  if ( *v23 )
  {
    v24 = v23[1];
    v25 = v23[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v25 = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = *(_QWORD *)(v24 + 16) & 3LL | v25;
  }
  *v23 = v18;
  if ( v18 )
  {
    v26 = *(_QWORD *)(v18 + 8);
    v23[1] = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = (unsigned __int64)(v23 + 1) | *(_QWORD *)(v26 + 16) & 3LL;
    v23[2] = (v18 + 8) | v23[2] & 3;
    *(_QWORD *)(v18 + 8) = v23;
  }
  v27 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
    v28 = *(_QWORD *)(v12 - 8);
  else
    v28 = v30 - 24 * v27;
  *(_QWORD *)(v28 + 8LL * (unsigned int)(v27 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = a5;
  return v12;
}
