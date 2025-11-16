// Function: sub_1707470
// Address: 0x1707470
//
__int64 __fastcall sub_1707470(__int64 a1, _BYTE *a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v6; // rax

  v6 = *(_QWORD *)(a3 + 8);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    return 0;
  else
    return sub_1707160(a1, a2, a3, a4, a5, a6);
}
