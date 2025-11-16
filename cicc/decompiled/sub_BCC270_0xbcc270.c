// Function: sub_BCC270
// Address: 0xbcc270
//
unsigned __int64 __fastcall sub_BCC270(__int64 *a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v5; // r8
  __int64 v6; // rsi
  _QWORD v8[3]; // [rsp+0h] [rbp-A0h] BYREF
  __int128 v9; // [rsp+18h] [rbp-88h]
  __int128 v10; // [rsp+28h] [rbp-78h]
  __int64 v11; // [rsp+38h] [rbp-68h]
  __int64 v12; // [rsp+40h] [rbp-60h]
  __int64 v13; // [rsp+48h] [rbp-58h]
  __int64 v14; // [rsp+50h] [rbp-50h]
  __int64 v15; // [rsp+58h] [rbp-48h]
  __int64 v16; // [rsp+60h] [rbp-40h]
  __int64 v17; // [rsp+68h] [rbp-38h]
  __int64 v18; // [rsp+70h] [rbp-30h]
  __int64 (__fastcall *v19)(); // [rsp+78h] [rbp-28h]

  v5 = *a1;
  v9 = 0;
  v6 = a1[1];
  v10 = 0;
  v19 = sub_C64CA0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v8[0] = sub_C94880(v5, v6);
  v8[1] = *a2;
  v8[2] = *a3;
  return sub_AC25F0(v8, 0x18u, (__int64)sub_C64CA0);
}
