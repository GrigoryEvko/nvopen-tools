// Function: sub_11D96D0
// Address: 0x11d96d0
//
__int64 *__fastcall sub_11D96D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int8 *v5; // rax

  v3 = *(_QWORD *)(a3 + 40);
  if ( !(unsigned __int8)sub_11D3030(a1, v3) )
    return (__int64 *)sub_F507F0(a3);
  v5 = (unsigned __int8 *)sub_11D6F60(a1, v3);
  return sub_B59720(a3, a2, v5);
}
