// Function: sub_14A6B70
// Address: 0x14a6b70
//
__int64 __fastcall sub_14A6B70(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // [rsp-48h] [rbp-48h] BYREF
  __int64 v19; // [rsp-40h] [rbp-40h]
  __int64 v20; // [rsp-38h] [rbp-38h]
  __int64 v21; // [rsp-30h] [rbp-30h]

  if ( !a1 )
    return 0;
  if ( *(_DWORD *)(a1 + 8) <= 1u )
    return 0;
  v2 = *(_QWORD *)(a1 + 16);
  v3 = (_QWORD *)(v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v2 & 4) != 0 )
    v3 = (_QWORD *)*v3;
  v4 = sub_1644900(v3, 64);
  v5 = sub_15A0680(v4, 0, 0);
  v8 = sub_1624210(v5, 0, v6, v7);
  v9 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v9 > 2 && (unsigned __int8)(**(_BYTE **)(a1 - 8 * v9) - 4) <= 0x1Eu )
  {
    v13 = sub_15A0680(v4, -1, 0);
    v18 = a1;
    v21 = sub_1624210(v13, -1, v14, v15);
    v16 = *(_QWORD *)(a1 + 16);
    v19 = a1;
    v20 = v8;
    v17 = (_QWORD *)(v16 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v16 & 4) != 0 )
      v17 = (_QWORD *)*v17;
    return sub_1627350(v17, &v18, 4, 0, 1);
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 16);
    v18 = a1;
    v19 = a1;
    v20 = v8;
    v11 = (_QWORD *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v10 & 4) != 0 )
      v11 = (_QWORD *)*v11;
    return sub_1627350(v11, &v18, 3, 0, 1);
  }
}
