// Function: sub_2ADF220
// Address: 0x2adf220
//
void __fastcall sub_2ADF220(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v4; // [rsp+8h] [rbp-18h] BYREF

  v4 = (unsigned __int8 *)a2;
  if ( *(_BYTE *)a2 > 0x1Cu
    && (unsigned __int8)sub_B19060(*(_QWORD *)(*(_QWORD *)*a1 + 416LL) + 56LL, *(_QWORD *)(a2 + 40), a3, a4)
    && !(unsigned __int8)sub_2AB37C0(a1[1], v4) )
  {
    sub_2ADF0D0(a1[2], (__int64 *)&v4);
  }
}
