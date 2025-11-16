// Function: sub_C4ECF0
// Address: 0xc4ecf0
//
unsigned __int64 __fastcall sub_C4ECF0(int *a1, __int64 *a2)
{
  int v2; // eax
  __int64 v3; // rax
  int v5; // [rsp+0h] [rbp-80h] BYREF
  _BYTE v6[20]; // [rsp+4h] [rbp-7Ch]
  __int128 v7; // [rsp+18h] [rbp-68h]
  __int128 v8; // [rsp+28h] [rbp-58h]
  __int64 v9; // [rsp+38h] [rbp-48h]
  __int64 v10; // [rsp+40h] [rbp-40h]
  __int64 v11; // [rsp+48h] [rbp-38h]
  __int64 v12; // [rsp+50h] [rbp-30h]
  __int64 v13; // [rsp+58h] [rbp-28h]
  __int64 v14; // [rsp+60h] [rbp-20h]
  __int64 v15; // [rsp+68h] [rbp-18h]
  __int64 v16; // [rsp+70h] [rbp-10h]
  __int64 (__fastcall *v17)(); // [rsp+78h] [rbp-8h]

  v2 = *a1;
  *(_OWORD *)&v6[4] = 0;
  v5 = v2;
  v3 = *a2;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = sub_C64CA0;
  *(_QWORD *)v6 = v3;
  v7 = 0;
  v8 = 0;
  return sub_AC25F0(&v5, 0xCu, (__int64)sub_C64CA0);
}
