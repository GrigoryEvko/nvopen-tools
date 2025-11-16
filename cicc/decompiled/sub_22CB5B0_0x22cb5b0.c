// Function: sub_22CB5B0
// Address: 0x22cb5b0
//
__int64 __fastcall sub_22CB5B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD v5[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v6)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v7)(__int64, __int64 *, __int64, __int64 *); // [rsp+18h] [rbp-18h]

  v7 = sub_22BDB40;
  v6 = sub_22BDB10;
  v5[0] = a3;
  sub_22CB1D0(a1, a2, a3, a4, (unsigned __int64)v5);
  if ( v6 )
    v6(v5, v5, 3);
  return a1;
}
