// Function: sub_BCC1E0
// Address: 0xbcc1e0
//
unsigned __int64 __fastcall sub_BCC1E0(__int64 *a1, __int64 *a2, _BYTE *a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  _QWORD v6[2]; // [rsp+0h] [rbp-80h] BYREF
  __int128 v7; // [rsp+10h] [rbp-70h]
  __int128 v8; // [rsp+20h] [rbp-60h]
  __int128 v9; // [rsp+30h] [rbp-50h]
  __int64 v10; // [rsp+40h] [rbp-40h]
  __int64 v11; // [rsp+48h] [rbp-38h]
  __int64 v12; // [rsp+50h] [rbp-30h]
  __int64 v13; // [rsp+58h] [rbp-28h]
  __int64 v14; // [rsp+60h] [rbp-20h]
  __int64 v15; // [rsp+68h] [rbp-18h]
  __int64 v16; // [rsp+70h] [rbp-10h]
  __int64 (__fastcall *v17)(); // [rsp+78h] [rbp-8h]

  v3 = *a1;
  v7 = 0;
  v6[0] = v3;
  v4 = *a2;
  v10 = 0;
  v6[1] = v4;
  LOBYTE(v4) = *a3;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = sub_C64CA0;
  LOBYTE(v7) = v4;
  v8 = 0;
  v9 = 0;
  return sub_AC25F0(v6, 0x11u, (__int64)sub_C64CA0);
}
