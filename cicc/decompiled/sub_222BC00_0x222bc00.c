// Function: sub_222BC00
// Address: 0x222bc00
//
void __fastcall sub_222BC00(__int64 a1)
{
  __int64 v1; // rax

  if ( !*(_BYTE *)(a1 + 168) && !*(_QWORD *)(a1 + 152) )
  {
    v1 = sub_2207820(*(_QWORD *)(a1 + 160));
    *(_BYTE *)(a1 + 168) = 1;
    *(_QWORD *)(a1 + 152) = v1;
  }
}
