// Function: sub_20A1B30
// Address: 0x20a1b30
//
__int64 __fastcall sub_20A1B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rax
  bool v7; // bl
  __int64 (*v9)(); // rax
  __int64 v10; // [rsp+8h] [rbp-98h] BYREF
  __m128i v11; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v12; // [rsp+28h] [rbp-78h]

  v10 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 112LL);
  sub_1562F70(&v11, v10, 0);
  v6 = sub_1560700(&v11, 20);
  v7 = sub_1560CB0(v6);
  sub_20A1200(v12);
  if ( v7 )
    return 0;
  if ( (unsigned __int8)sub_1560260(&v10, 0, 58) )
    return 0;
  if ( (unsigned __int8)sub_1560260(&v10, 0, 40) )
    return 0;
  v9 = *(__int64 (**)())(*(_QWORD *)a1 + 1232LL);
  if ( v9 == sub_20A0810 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64))v9)(a1, a3, a4);
}
