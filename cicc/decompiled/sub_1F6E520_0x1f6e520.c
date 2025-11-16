// Function: sub_1F6E520
// Address: 0x1f6e520
//
__int64 __fastcall sub_1F6E520(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  unsigned int v8; // r13d
  __int64 v10; // r12
  __int64 v11; // r12
  _DWORD *v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  bool v16; // r12
  bool v17; // al
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _BYTE v27[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v28; // [rsp+28h] [rbp-38h]

  v7 = *(unsigned __int16 *)(a2 + 24);
  if ( v7 == 137 )
  {
    v22 = *(_QWORD *)(a2 + 32);
    v8 = 1;
    *(_QWORD *)a4 = *(_QWORD *)v22;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v22 + 8);
    v23 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)a5 = *(_QWORD *)(v23 + 40);
    *(_DWORD *)(a5 + 8) = *(_DWORD *)(v23 + 48);
    v24 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)a6 = *(_QWORD *)(v24 + 80);
    *(_DWORD *)(a6 + 8) = *(_DWORD *)(v24 + 88);
    return v8;
  }
  if ( v7 != 136 )
    return 0;
  v10 = a3;
  if ( !(unsigned __int8)sub_20ABB20(*(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL)) )
    return 0;
  v8 = sub_20ABF20(*(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL));
  if ( !(_BYTE)v8 )
    return 0;
  v11 = *(_QWORD *)(a2 + 40) + 16 * v10;
  v12 = *(_DWORD **)(a1 + 8);
  v13 = *(_QWORD *)(v11 + 8);
  v14 = a4;
  v15 = a5;
  v27[0] = *(_BYTE *)v11;
  v28 = v13;
  if ( !v27[0] )
  {
    v16 = sub_1F58CD0((__int64)v27);
    v17 = sub_1F58D20((__int64)v27);
    v14 = a4;
    v15 = a5;
    if ( !v17 )
      goto LABEL_9;
LABEL_16:
    v18 = v12[17];
    goto LABEL_11;
  }
  if ( (unsigned __int8)(v27[0] - 14) <= 0x5Fu )
    goto LABEL_16;
  v16 = (unsigned __int8)(v27[0] - 86) <= 0x17u || (unsigned __int8)(v27[0] - 8) <= 5u;
LABEL_9:
  if ( v16 )
    v18 = v12[16];
  else
    v18 = v12[15];
LABEL_11:
  if ( v18 )
  {
    v19 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)v14 = *(_QWORD *)v19;
    *(_DWORD *)(v14 + 8) = *(_DWORD *)(v19 + 8);
    v20 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)v15 = *(_QWORD *)(v20 + 40);
    *(_DWORD *)(v15 + 8) = *(_DWORD *)(v20 + 48);
    v21 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)a6 = *(_QWORD *)(v21 + 160);
    *(_DWORD *)(a6 + 8) = *(_DWORD *)(v21 + 168);
    return v8;
  }
  return 0;
}
