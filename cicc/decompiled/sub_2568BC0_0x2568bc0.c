// Function: sub_2568BC0
// Address: 0x2568bc0
//
__int64 __fastcall sub_2568BC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int64 v3; // rax
  __int64 v4; // rax

  v2 = *(_QWORD *)(a2 + 208);
  v3 = sub_250C680((__int64 *)(a1 + 72));
  v4 = sub_251B1C0(v2, *(_QWORD *)(v3 + 24));
  if ( *(_BYTE *)(v4 + 112) || *(_BYTE *)(v4 + 113) )
    return 1;
  else
    return sub_255A010(a1, a2);
}
