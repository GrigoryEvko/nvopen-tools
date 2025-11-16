// Function: sub_1D47530
// Address: 0x1d47530
//
__int64 __fastcall sub_1D47530(__int64 a1)
{
  _QWORD *v1; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 8;
  v1 = (_QWORD *)malloc(8u);
  if ( !v1 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v1 = 0;
  }
  *(_QWORD *)a1 = v1;
  *v1 = 0;
  *(_QWORD *)(a1 + 8) = 1;
  return a1;
}
