// Function: sub_8DBF30
// Address: 0x8dbf30
//
_BOOL8 __fastcall sub_8DBF30(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 168);
  return v1 && *(_BYTE *)(v1 + 173) == 12 || (unsigned int)sub_8DBE70(*(_QWORD *)(a1 + 160)) != 0;
}
