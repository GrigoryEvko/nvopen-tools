// Function: sub_A5BF40
// Address: 0xa5bf40
//
void __fastcall sub_A5BF40(unsigned __int8 *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // r12
  bool v7; // al
  __int64 v8; // [rsp+8h] [rbp-238h]
  __int64 v9; // [rsp+8h] [rbp-238h]
  _QWORD v10[14]; // [rsp+10h] [rbp-230h] BYREF
  _BYTE v11[448]; // [rsp+80h] [rbp-1C0h] BYREF

  v4 = a2;
  v5 = a4;
  if ( !a4 )
    v5 = sub_A4F760(a1);
  if ( a3 || (v8 = v4, v7 = sub_A5BC40(a1, v4, 0, v5), v4 = v8, !v7) )
  {
    v9 = v4;
    sub_A55A10((__int64)v11, v5, *a1 == 24);
    sub_A55860((__int64)v10, (__int64)v11, v5, 0);
    sub_A5BCB0((__int64)a1, v9, a3, (__int64)v10);
    sub_A55520(v10, v9);
    sub_A552A0((__int64)v11, v9);
  }
}
