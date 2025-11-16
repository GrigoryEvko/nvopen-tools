// Function: sub_A4F6A0
// Address: 0xa4f6a0
//
__int64 __fastcall sub_A4F6A0(__int64 a1)
{
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 40) )
  {
    sub_C66990(a1, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 32) - *(_QWORD *)(a1 + 16));
    *(_BYTE *)(a1 + 104) = 1;
    sub_CB6DC0(a1);
    v2 = *(_QWORD *)(a1 + 32);
    *(_BYTE *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 64) = v2;
  }
  return a1;
}
