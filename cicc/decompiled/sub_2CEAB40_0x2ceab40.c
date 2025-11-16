// Function: sub_2CEAB40
// Address: 0x2ceab40
//
unsigned __int64 __fastcall sub_2CEAB40(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v8; // rax
  __int64 **v9; // rax
  _QWORD *v12; // r8
  unsigned __int8 v13; // r9
  int v14; // r10d
  _QWORD *v15; // r11
  __int64 *v16; // rax
  __int64 **v17; // rax

  if ( *a3 == 20 || sub_2CDF320((__int64)a3) )
  {
    v8 = (__int64 *)sub_BD5C60((__int64)a3);
    v9 = (__int64 **)sub_BCE3C0(v8, a6);
    return sub_AC9EC0(v9);
  }
  else if ( (unsigned int)(v14 - 12) > 1 )
  {
    return sub_2CE9A90(v15, a2, a3, a4, v12, a6, v13);
  }
  else
  {
    v16 = (__int64 *)sub_BD5C60((__int64)a3);
    v17 = (__int64 **)sub_BCE3C0(v16, a6);
    return sub_ACA8A0(v17);
  }
}
