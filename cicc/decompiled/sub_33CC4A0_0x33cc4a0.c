// Function: sub_33CC4A0
// Address: 0x33cc4a0
//
__int64 __fastcall sub_33CC4A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  _QWORD v10[4]; // [rsp+0h] [rbp-20h] BYREF

  v10[0] = a2;
  v6 = *(__int64 **)(a1 + 64);
  v10[1] = a3;
  if ( (_WORD)a2 == 510 )
    v7 = sub_BCE3C0(v6, 0);
  else
    v7 = sub_3007410((__int64)v10, v6, a3, a4, a5, a6);
  v8 = sub_2E79000(*(__int64 **)(a1 + 40));
  return sub_AE5020(v8, v7);
}
