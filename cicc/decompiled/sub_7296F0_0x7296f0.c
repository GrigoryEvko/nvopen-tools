// Function: sub_7296F0
// Address: 0x7296f0
//
_DWORD *__fastcall sub_7296F0(int a1, _DWORD *a2)
{
  int v2; // edi
  _DWORD *result; // rax

  v2 = *(_DWORD *)(qword_4F04C68[0] + 776LL * a1 + 192);
  result = (_DWORD *)dword_4F07270[0];
  if ( dword_4F07270[0] == v2 )
  {
    *a2 = 0;
  }
  else
  {
    *a2 = dword_4F07270[0];
    return sub_7296B0(v2);
  }
  return result;
}
