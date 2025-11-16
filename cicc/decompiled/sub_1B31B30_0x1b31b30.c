// Function: sub_1B31B30
// Address: 0x1b31b30
//
void __fastcall sub_1B31B30(__int64 a1, __int64 ***a2)
{
  __int64 *v3; // rax
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 **v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // r12
  __int64 v17; // r10
  __int64 *v18; // [rsp+0h] [rbp-70h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  _QWORD *v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v22[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v23; // [rsp+30h] [rbp-40h]

  v3 = (__int64 *)sub_15F2050((__int64)a2);
  v4 = sub_15E26F0(v3, 4, 0, 0);
  v7 = sub_15A06D0(*a2, 4, v5, v6);
  v23 = 257;
  v8 = v7;
  v9 = sub_1648A60(56, 2u);
  v10 = (__int64)v9;
  if ( v9 )
  {
    v19 = (__int64)v9;
    v11 = *a2;
    if ( *((_BYTE *)*a2 + 8) == 16 )
    {
      v18 = v11[4];
      v12 = (__int64 *)sub_1643320(*v11);
      v13 = (__int64)sub_16463B0(v12, (unsigned int)v18);
    }
    else
    {
      v13 = sub_1643320(*v11);
    }
    sub_15FEC10(v10, v13, 51, 33, (__int64)a2, v8, (__int64)v22, 0);
    sub_15F2180(v19, (__int64)a2);
  }
  else
  {
    sub_15F2180(0, (__int64)a2);
  }
  v21 = v10;
  v23 = 257;
  v14 = *(_QWORD *)(*(_QWORD *)v4 + 24LL);
  v15 = sub_1648AB0(72, 2u, 0);
  v16 = (__int64)v15;
  if ( v15 )
  {
    v20 = v15;
    sub_15F1EA0((__int64)v15, **(_QWORD **)(v14 + 16), 54, (__int64)(v15 - 6), 2, 0);
    *(_QWORD *)(v16 + 56) = 0;
    sub_15F5B40(v16, v14, v4, &v21, 1, (__int64)v22, 0, 0);
    v17 = (__int64)v20;
  }
  else
  {
    v17 = 0;
  }
  sub_15F2180(v17, v10);
  sub_14CE830(a1, v16);
}
