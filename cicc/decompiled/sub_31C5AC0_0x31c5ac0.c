// Function: sub_31C5AC0
// Address: 0x31c5ac0
//
__int64 __fastcall sub_31C5AC0(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  unsigned int v9; // esi
  unsigned __int64 v11; // [rsp+8h] [rbp-38h] BYREF
  __int64 v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v13; // [rsp+18h] [rbp-28h]
  __int64 v14[4]; // [rsp+20h] [rbp-20h] BYREF

  v12 = a2;
  v13 = a3;
  v5 = sub_C93460(&v12, "_", 1u);
  v6 = v13;
  v7 = 0;
  v8 = v5 + 1;
  if ( v8 <= v13 )
  {
    v6 = v8;
    v7 = v13 - v8;
  }
  v14[1] = v7;
  v14[0] = v12 + v6;
  if ( (unsigned __int8)sub_C93B20(v14, 0xAu, &v11) || (v9 = v11, v11 != (unsigned int)v11) )
    v9 = 52;
  return sub_31C5850(a1, v9, a4);
}
