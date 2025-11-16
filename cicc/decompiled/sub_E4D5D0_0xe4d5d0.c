// Function: sub_E4D5D0
// Address: 0xe4d5d0
//
__int64 __fastcall sub_E4D5D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v6; // rax
  __int64 v7; // [rsp+8h] [rbp-18h]

  v4 = *(_QWORD *)(a1 + 8);
  if ( !a4 )
  {
    v7 = a3;
    v6 = sub_E92830(*(_QWORD *)(*(_QWORD *)(v4 + 168) + 24LL), *(_QWORD *)(a1 + 8));
    a3 = v7;
    a4 = v6;
  }
  return sub_E4D1D0(a1, 0x7FFFFFFFFFFFFFFFLL, a3, a4, *(_DWORD *)(*(_QWORD *)(v4 + 152) + 8LL));
}
