// Function: sub_27FF4A0
// Address: 0x27ff4a0
//
__int64 __fastcall sub_27FF4A0(__int64 a1, __int64 *a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v7; // rax
  __int64 v9; // rdi
  unsigned int v11; // r15d
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // r14
  unsigned int v17; // r12d
  bool v18; // al
  unsigned int v19; // r12d
  __int64 v20; // r8
  unsigned __int64 v21; // rax
  _BYTE *v22; // r12
  unsigned int v23; // r15d
  __int64 v24; // r13
  unsigned __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // r13
  bool v29; // cc
  unsigned __int64 v30; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-58h]
  unsigned __int64 v32[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v33[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a3 + 48);
  if ( !v4 )
    return 0;
  if ( *(_QWORD *)(v4 + 40) != 2 )
    return 0;
  v7 = *(_QWORD *)(*(_QWORD *)(v4 + 32) + 8LL);
  if ( *(_WORD *)(v7 + 24) )
    return 0;
  v9 = *(_QWORD *)(v7 + 32);
  v5 = a4;
  v11 = *(_DWORD *)(v9 + 32);
  v12 = *(_QWORD *)(v9 + 24);
  v13 = 1LL << ((unsigned __int8)v11 - 1);
  if ( v11 <= 0x40 )
  {
    if ( (v13 & v12) == 0 && v12 )
      goto LABEL_9;
    return 0;
  }
  if ( (*(_QWORD *)(v12 + 8LL * ((v11 - 1) >> 6)) & v13) != 0 || v11 == (unsigned int)sub_C444A0(v9 + 24) )
    return 0;
LABEL_9:
  if ( (_BYTE)v5 )
  {
    v24 = sub_DBA6E0((__int64)a2, a1, *(_QWORD *)(*(_QWORD *)(a3 + 8) + 40LL), 0);
    if ( !sub_D96A50(v24) )
    {
      *(_QWORD *)(a3 + 56) = v24;
      return v5;
    }
    return 0;
  }
  v14 = *(_DWORD *)(a3 + 16);
  v5 = 1;
  if ( ((v14 - 36) & 0xFFFFFFFB) == 0 )
    return v5;
  if ( ((v14 - 37) & 0xFFFFFFFB) != 0 )
    return 0;
  v15 = sub_D95540(*(_QWORD *)(a3 + 56));
  v16 = v15;
  if ( *(_BYTE *)(v15 + 8) != 12 )
    return 0;
  v17 = *(_DWORD *)(v15 + 8);
  v18 = sub_B532B0(*(_DWORD *)(a3 + 16));
  v19 = v17 >> 8;
  v31 = v19;
  if ( v18 )
  {
    v20 = ~(1LL << ((unsigned __int8)v19 - 1));
    if ( v19 <= 0x40 )
    {
      v21 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
      if ( !v19 )
        v21 = 0;
      v30 = v21;
      goto LABEL_18;
    }
    sub_C43690((__int64)&v30, -1, 1);
    v20 = ~(1LL << ((unsigned __int8)v19 - 1));
    if ( v31 <= 0x40 )
    {
LABEL_18:
      v30 &= v20;
      goto LABEL_19;
    }
    *(_QWORD *)(v30 + 8LL * ((v19 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v19 - 1));
  }
  else if ( v19 > 0x40 )
  {
    sub_C43690((__int64)&v30, -1, 1);
  }
  else
  {
    v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
    if ( !v19 )
      v25 = 0;
    v30 = v25;
  }
LABEL_19:
  v22 = sub_DA26C0(a2, (__int64)&v30);
  v23 = !sub_B532B0(*(_DWORD *)(a3 + 16)) ? 36 : 40;
  v5 = sub_DC3A60((__int64)a2, v23, *(_BYTE **)(a3 + 56), v22);
  if ( !(_BYTE)v5 )
  {
    if ( v31 <= 0x40 )
      return v5;
    goto LABEL_21;
  }
  v26 = sub_DA2C50((__int64)a2, v16, 1, 0);
  v27 = *(_QWORD *)(a3 + 56);
  v33[1] = v26;
  v33[0] = v27;
  v32[0] = (unsigned __int64)v33;
  v32[1] = 0x200000002LL;
  v28 = sub_DC7EB0(a2, (__int64)v32, 0, 0);
  if ( (_QWORD *)v32[0] != v33 )
    _libc_free(v32[0]);
  v29 = v31 <= 0x40;
  *(_QWORD *)(a3 + 56) = v28;
  *(_DWORD *)(a3 + 16) = v23;
  *(_BYTE *)(a3 + 20) = 0;
  if ( !v29 )
  {
LABEL_21:
    if ( v30 )
      j_j___libc_free_0_0(v30);
  }
  return v5;
}
