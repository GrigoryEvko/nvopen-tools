// Function: sub_72B8C0
// Address: 0x72b8c0
//
void __fastcall sub_72B8C0(__int64 a1)
{
  bool v1; // zf

  sub_72B850(a1);
  *(_BYTE *)(a1 + 89) &= 0xFAu;
  v1 = *(_QWORD *)(a1 + 40) == 0;
  *(_QWORD *)(a1 + 48) = 0;
  if ( !v1 )
    *(_QWORD *)(a1 + 40) = unk_4F07288;
}
