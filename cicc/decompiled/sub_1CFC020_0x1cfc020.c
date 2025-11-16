// Function: sub_1CFC020
// Address: 0x1cfc020
//
__int64 __fastcall sub_1CFC020(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(720);
  v2 = v1;
  if ( v1 )
  {
    sub_1D0DA30(v1, *(_QWORD *)(a1 + 256));
    *(_QWORD *)(v2 + 664) = 0;
    *(_QWORD *)v2 = off_49F9420;
    *(_QWORD *)(v2 + 672) = 0;
    *(_QWORD *)(v2 + 680) = 0;
    *(_QWORD *)(v2 + 688) = 0;
    *(_QWORD *)(v2 + 696) = 0;
    *(_QWORD *)(v2 + 704) = 0;
    *(_DWORD *)(v2 + 712) = 0;
  }
  return v2;
}
