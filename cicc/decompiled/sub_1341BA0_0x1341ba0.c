// Function: sub_1341BA0
// Address: 0x1341ba0
//
__int64 __fastcall sub_1341BA0(__int64 a1, __int64 a2, unsigned __int64 *a3, unsigned int a4, unsigned __int8 a5)
{
  _QWORD *v6; // rdx
  unsigned int v9; // eax
  unsigned int v10; // r11d
  __int64 *v12; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 *v13; // [rsp+18h] [rbp-1B8h] BYREF
  _QWORD v14[54]; // [rsp+20h] [rbp-1B0h] BYREF

  v6 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    sub_130D500(v14);
    v6 = v14;
  }
  LOBYTE(v9) = sub_1341260(a1, a2, v6, (__int64)a3, 0, 1, (__int64 *)&v12, (unsigned __int64 *)&v13);
  v10 = v9;
  if ( !(_BYTE)v9 )
    sub_1341500(v12, v13, a3, a4, a5);
  return v10;
}
