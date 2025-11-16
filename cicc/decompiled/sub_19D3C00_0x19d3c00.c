// Function: sub_19D3C00
// Address: 0x19d3c00
//
__int64 __fastcall sub_19D3C00(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v7; // rbx

  v7 = sub_1649C60(*(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
  if ( v7 == sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) )
    return sub_19D3100(a1, a2, a3, a4, a5, a6);
  else
    return 0;
}
