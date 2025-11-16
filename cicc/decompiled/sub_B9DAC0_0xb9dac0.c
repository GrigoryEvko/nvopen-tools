// Function: sub_B9DAC0
// Address: 0xb9dac0
//
__int64 __fastcall sub_B9DAC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r13
  unsigned __int8 v5; // al
  _BYTE **v6; // rdx
  _BYTE *v7; // rdi
  unsigned __int8 v8; // al
  _QWORD *v9; // rdx
  _BYTE *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v15; // r12d
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  bool v24; // cc
  _QWORD *v25; // rdx
  _QWORD *v26; // rax
  unsigned __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // [rsp+0h] [rbp-70h]
  _BYTE *v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+28h] [rbp-48h] BYREF
  __int64 v37[8]; // [rsp+30h] [rbp-40h] BYREF

  v36 = sub_BD5C60(a3, a2);
  v4 = (__int64 *)v36;
  v33 = (_BYTE *)(a1 - 16);
  v5 = *(_BYTE *)(a1 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(_BYTE ***)(a1 - 32);
  else
    v6 = (_BYTE **)&v33[-8 * ((v5 >> 2) & 0xF)];
  v7 = *v6;
  if ( **v6 )
    v7 = 0;
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(_QWORD **)(a2 - 32);
  else
    v9 = (_QWORD *)(a2 - 16 - 8LL * ((v8 >> 2) & 0xF));
  v10 = (_BYTE *)*v9;
  if ( *(_BYTE *)*v9 )
    v10 = 0;
  v34 = (__int64)v10;
  v32 = sub_B91420((__int64)v7);
  v35 = v11;
  v12 = sub_B91420(v34);
  if ( v35 != 14
    || *(_QWORD *)v32 != 0x775F68636E617262LL
    || *(_DWORD *)(v32 + 8) != 1751607653
    || *(_WORD *)(v32 + 12) != 29556
    || v13 != 14
    || *(_QWORD *)v12 != 0x775F68636E617262LL
    || *(_DWORD *)(v12 + 8) != 1751607653
    || *(_WORD *)(v12 + 12) != 29556 )
  {
    return 0;
  }
  v15 = sub_BC8810(a1);
  v16 = sub_A17150(v33);
  v17 = v15;
  v18 = 0;
  v19 = *(_QWORD *)&v16[8 * v17];
  if ( *(_BYTE *)v19 == 1 )
  {
    v18 = *(_QWORD *)(v19 + 136);
    if ( *(_BYTE *)v18 != 17 )
      v18 = 0;
  }
  v20 = (unsigned int)sub_BC8810(a2);
  v21 = *(_QWORD *)&sub_A17150((_BYTE *)(a2 - 16))[8 * v20];
  if ( *(_BYTE *)v21 != 1 || (v22 = *(_QWORD *)(v21 + 136), *(_BYTE *)v22 != 17) )
  {
    v37[0] = sub_B8C130(&v36, (__int64)"branch_weights", 14);
    BUG();
  }
  v23 = sub_B8C130(&v36, (__int64)"branch_weights", 14);
  v24 = *(_DWORD *)(v22 + 32) <= 0x40u;
  v37[0] = v23;
  if ( v24 )
    v25 = *(_QWORD **)(v22 + 24);
  else
    v25 = **(_QWORD ***)(v22 + 24);
  v26 = *(_QWORD **)(v18 + 24);
  if ( *(_DWORD *)(v18 + 32) > 0x40u )
    v26 = (_QWORD *)*v26;
  v27 = (unsigned __int64)v25 + (_QWORD)v26;
  if ( v26 < v25 )
    v26 = v25;
  if ( v27 < (unsigned __int64)v26 )
    v27 = -1;
  v28 = sub_BCB2E0(v4);
  v29 = sub_ACD640(v28, v27, 0);
  v37[1] = sub_B8C140((__int64)&v36, v29, v30, v31);
  return sub_B9C770(v4, v37, (__int64 *)2, 0, 1);
}
