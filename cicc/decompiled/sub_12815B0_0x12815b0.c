// Function: sub_12815B0
// Address: 0x12815b0
//
__int64 __fastcall sub_12815B0(__int64 *a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  bool v8; // cc
  _QWORD *v9; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r10
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 *v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned int v23; // [rsp+Ch] [rbp-84h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  _QWORD v27[2]; // [rsp+28h] [rbp-68h] BYREF
  __int64 v28; // [rsp+38h] [rbp-58h] BYREF
  _BYTE v29[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v30; // [rsp+50h] [rbp-40h]

  v6 = a2;
  v8 = a3[16] <= 0x10u;
  v27[0] = a4;
  if ( v8 && *(_BYTE *)(a4 + 16) <= 0x10u )
  {
    v28 = a4;
    v29[4] = 0;
    return sub_15A2E80(a2, (_DWORD)a3, (unsigned int)&v28, 1, 0, (unsigned int)v29, 0);
  }
  else
  {
    v30 = 257;
    if ( !a2 )
    {
      v22 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v22 = **(_QWORD **)(v22 + 16);
      v6 = *(_QWORD *)(v22 + 24);
    }
    v11 = sub_1648A60(72, 2);
    v9 = (_QWORD *)v11;
    if ( v11 )
    {
      v25 = v11;
      v24 = v11 - 48;
      v12 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v12 = **(_QWORD **)(v12 + 16);
      v23 = *(_DWORD *)(v12 + 8) >> 8;
      v13 = sub_15F9F50(v6, v27, 1);
      v14 = sub_1646BA0(v13, v23);
      v15 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 || (v15 = *(_QWORD *)v27[0], *(_BYTE *)(*(_QWORD *)v27[0] + 8LL) == 16) )
        v14 = sub_16463B0(v14, *(_QWORD *)(v15 + 32));
      sub_15F1EA0(v9, v14, 32, v24, 2, 0);
      v9[7] = v6;
      v9[8] = sub_15F9F50(v6, v27, 1);
      sub_15F9CE0(v9, a3, v27, 1, v29);
    }
    else
    {
      v25 = 0;
    }
    v16 = a1[1];
    if ( v16 )
    {
      v17 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v16 + 40, v9);
      v18 = v9[3];
      v19 = *v17;
      v9[4] = v17;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      v9[3] = v19 | v18 & 7;
      *(_QWORD *)(v19 + 8) = v9 + 3;
      *v17 = *v17 & 7 | (unsigned __int64)(v9 + 3);
    }
    sub_164B780(v25, a5);
    v20 = *a1;
    if ( *a1 )
    {
      v28 = *a1;
      sub_1623A60(&v28, v20, 2);
      if ( v9[6] )
        sub_161E7C0(v9 + 6);
      v21 = v28;
      v9[6] = v28;
      if ( v21 )
        sub_1623210(&v28, v21, v9 + 6);
    }
  }
  return (__int64)v9;
}
