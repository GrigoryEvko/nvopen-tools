// Function: sub_1413D80
// Address: 0x1413d80
//
__int64 __fastcall sub_1413D80(__int64 a1)
{
  bool v1; // zf

  v1 = *(_BYTE *)(a1 + 1104) == 0;
  *(_QWORD *)a1 = &unk_49EB110;
  if ( !v1 )
    sub_14139D0(a1 + 160);
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
