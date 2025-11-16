// Function: sub_A4F5A0
// Address: 0xa4f5a0
//
__int64 __fastcall sub_A4F5A0(__int64 a1, unsigned int a2, unsigned __int8 a3, unsigned __int8 a4)
{
  __int64 v7; // rax

  if ( *(_BYTE *)(a1 + 40) )
  {
    sub_C66990(a1, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 32) - *(_QWORD *)(a1 + 16));
    *(_BYTE *)(a1 + 104) = 1;
    sub_CB6D50(a1, a2, a3, a4);
    v7 = *(_QWORD *)(a1 + 32);
    *(_BYTE *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 64) = v7;
  }
  return a1;
}
