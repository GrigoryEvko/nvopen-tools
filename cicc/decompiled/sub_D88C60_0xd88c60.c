// Function: sub_D88C60
// Address: 0xd88c60
//
__int64 __fastcall sub_D88C60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned int v10; // r12d
  _QWORD *v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int16 v16; // ax
  unsigned __int8 v17; // r9
  unsigned __int16 v20; // ax
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+28h] [rbp-58h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  unsigned __int8 v26; // [rsp+28h] [rbp-58h]
  __int64 v27[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28[8]; // [rsp+40h] [rbp-40h] BYREF

  v7 = sub_D85880(a1, a2);
  v8 = sub_D85880(a1, a4);
  if ( v7 && v8 && (v9 = sub_DCC810(*(_QWORD *)(a1 + 16), v7, v8, 0, 0), !(unsigned __int8)sub_D96A50(v9)) )
  {
    sub_D882F0(v27, a4);
    v10 = *(_DWORD *)(a1 + 24);
    v11 = (_QWORD *)sub_B2BE50(**(_QWORD **)(a1 + 16));
    v12 = sub_BCD140(v11, v10);
    v13 = sub_DA26C0(*(_QWORD *)(a1 + 16), v27);
    v22 = sub_DC5760(*(_QWORD *)(a1 + 16), v13, v12, 0);
    v21 = *(_QWORD *)(a1 + 16);
    v24 = sub_DC5760(v21, a5, v12, 0);
    v14 = sub_DA26C0(*(_QWORD *)(a1 + 16), v28);
    v15 = sub_DC5760(*(_QWORD *)(a1 + 16), v14, v12, 0);
    v25 = sub_DCC810(v21, v15, v24, 0, 0);
    v16 = sub_DDCA80(*(_QWORD *)(a1 + 16), 39, v9, v22, a3);
    v17 = 0;
    if ( HIBYTE(v16) )
    {
      if ( (_BYTE)v16 )
      {
        v20 = sub_DDCA80(*(_QWORD *)(a1 + 16), 41, v9, v25, a3);
        v17 = 0;
        if ( HIBYTE(v20) )
          v17 = v20;
      }
    }
    v26 = v17;
    sub_969240(v28);
    sub_969240(v27);
    return v26;
  }
  else
  {
    return 0;
  }
}
