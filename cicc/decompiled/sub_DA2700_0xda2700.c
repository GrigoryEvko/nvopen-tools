// Function: sub_DA2700
// Address: 0xda2700
//
_QWORD *__fastcall sub_DA2700(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  bool v4; // zf
  __int64 v6; // rsi
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ebx
  unsigned int v11; // eax
  unsigned int v12; // r8d
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 *v16; // r15
  unsigned int v17; // eax
  _QWORD *v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-68h]
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-58h]
  _QWORD *v27; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-48h]
  _QWORD *v29; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-38h]

  v3 = (_QWORD *)a2;
  v4 = *(_WORD *)(a2 + 24) == 0;
  v22 = 1;
  v21 = 0;
  v24 = 1;
  v23 = 0;
  if ( !v4 || *(_WORD *)(a3 + 24) )
    return v3;
  v6 = *(_QWORD *)(a2 + 32);
  if ( *(_DWORD *)(v6 + 32) <= 0x40u )
  {
    v22 = *(_DWORD *)(v6 + 32);
    v8 = *(_QWORD *)(a3 + 32);
    v19 = *(_QWORD **)(v6 + 24);
    v10 = *(_DWORD *)(v8 + 32);
    v9 = v8 + 24;
    v21 = v19;
    if ( v10 > 0x40 )
    {
LABEL_6:
      sub_C43990((__int64)&v23, v9);
      v10 = v24;
      goto LABEL_7;
    }
  }
  else
  {
    sub_C43990((__int64)&v21, v6 + 24);
    v8 = *(_QWORD *)(a3 + 32);
    v9 = v8 + 24;
    if ( v24 > 0x40 )
      goto LABEL_6;
    v10 = *(_DWORD *)(v8 + 32);
    if ( v10 > 0x40 )
      goto LABEL_6;
  }
  v20 = *(_QWORD *)(v8 + 24);
  v24 = v10;
  v23 = v20;
LABEL_7:
  v11 = v22;
  v12 = v22;
  v13 = (unsigned __int64)v21;
  if ( v22 > 0x40 )
    v13 = v21[(v22 - 1) >> 6];
  if ( (v13 & (1LL << ((unsigned __int8)v22 - 1))) != 0 )
    goto LABEL_25;
  v14 = v23;
  v15 = 1LL << ((unsigned __int8)v10 - 1);
  if ( v10 > 0x40 )
  {
    if ( (*(_QWORD *)(v23 + 8LL * ((v10 - 1) >> 6)) & v15) != 0 )
    {
LABEL_27:
      if ( v14 )
        j_j___libc_free_0_0(v14);
      v12 = v22;
      goto LABEL_30;
    }
    if ( (unsigned int)sub_C444A0((__int64)&v23) == v10 )
    {
LABEL_26:
      v14 = v23;
      goto LABEL_27;
    }
LABEL_13:
    sub_C4B490((__int64)&v25, (__int64)&v21, (__int64)&v23);
    v16 = *(__int64 **)(a1 + 8);
    v28 = v22;
    if ( v22 > 0x40 )
      sub_C43780((__int64)&v27, (const void **)&v21);
    else
      v27 = v21;
    sub_C46B40((__int64)&v27, &v25);
    v17 = v28;
    v28 = 0;
    v30 = v17;
    v29 = v27;
    v3 = sub_DA26C0(v16, (__int64)&v29);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    v10 = v24;
LABEL_25:
    if ( v10 > 0x40 )
      goto LABEL_26;
    v11 = v22;
    goto LABEL_41;
  }
  if ( (v15 & v23) == 0 )
  {
    if ( !v23 )
    {
LABEL_41:
      v12 = v11;
      goto LABEL_30;
    }
    goto LABEL_13;
  }
LABEL_30:
  if ( v12 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return v3;
}
