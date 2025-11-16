// Function: sub_2854010
// Address: 0x2854010
//
__int64 __fastcall sub_2854010(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // r13
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 *v11; // r14
  __int64 v12; // r13
  __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 *v21; // r13
  __int64 v22; // r12
  __int64 *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 *v26; // rdi
  __int64 result; // rax
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rdx

  v5 = sub_B98A20(a2, a2);
  v6 = (__int64 *)sub_BD5C60(a1);
  v7 = sub_B9F6F0(v6, v5);
  v8 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v8 )
  {
    v9 = *(_QWORD *)(v8 + 8);
    **(_QWORD **)(v8 + 16) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
  }
  *(_QWORD *)v8 = v7;
  if ( v7 )
  {
    v10 = *(_QWORD *)(v7 + 16);
    *(_QWORD *)(v8 + 8) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v8 + 8;
    *(_QWORD *)(v8 + 16) = v7 + 16;
    *(_QWORD *)(v7 + 16) = v8;
  }
  v11 = *(__int64 **)a3;
  v12 = *(unsigned int *)(a3 + 8);
  v13 = (__int64 *)sub_BD5C60(a1);
  v14 = sub_B0D000(v13, v11, v12, 0, 1);
  v15 = *(_QWORD *)(v14 + 8);
  v16 = (__int64 *)(v15 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v15 & 4) != 0 )
    v16 = (__int64 *)*v16;
  v17 = sub_B9F6F0(v16, (_BYTE *)v14);
  v18 = a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v18 )
  {
    v19 = *(_QWORD *)(v18 + 8);
    **(_QWORD **)(v18 + 16) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = *(_QWORD *)(v18 + 16);
  }
  *(_QWORD *)v18 = v17;
  if ( v17 )
  {
    v20 = *(_QWORD *)(v17 + 16);
    *(_QWORD *)(v18 + 8) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = v18 + 8;
    *(_QWORD *)(v18 + 16) = v17 + 16;
    *(_QWORD *)(v17 + 16) = v18;
  }
  v21 = *(__int64 **)a3;
  v22 = *(unsigned int *)(a3 + 8);
  v23 = (__int64 *)sub_BD5C60(a1);
  v24 = sub_B0D000(v23, v21, v22, 0, 1);
  v25 = *(_QWORD *)(v24 + 8);
  v26 = (__int64 *)(v25 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v25 & 4) != 0 )
    v26 = (__int64 *)*v26;
  result = sub_B9F6F0(v26, (_BYTE *)v24);
  v28 = 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + a1;
  if ( *(_QWORD *)v28 )
  {
    v29 = *(_QWORD *)(v28 + 8);
    **(_QWORD **)(v28 + 16) = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 16) = *(_QWORD *)(v28 + 16);
  }
  *(_QWORD *)v28 = result;
  if ( result )
  {
    v30 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v28 + 8) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = v28 + 8;
    *(_QWORD *)(v28 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v28;
  }
  return result;
}
