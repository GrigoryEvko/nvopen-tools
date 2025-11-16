// Function: sub_7296C0
// Address: 0x7296c0
//
_DWORD *__fastcall sub_7296C0(_DWORD *a1)
{
  _DWORD *result; // rax

  result = &unk_4F073B8;
  if ( dword_4F07270[0] == unk_4F073B8 )
  {
    *a1 = 0;
  }
  else
  {
    *a1 = dword_4F07270[0];
    return sub_7296B0(unk_4F073B8);
  }
  return result;
}
