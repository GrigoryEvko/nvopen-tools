// Function: sub_179D670
// Address: 0x179d670
//
unsigned __int64 __fastcall sub_179D670(_DWORD *a1, unsigned __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int64 result; // rax
  unsigned int v4; // r13d

  v2 = a1[2];
  if ( v2 > 0x40 )
  {
    v4 = v2 - sub_16A57B0((__int64)a1);
    result = a2;
    if ( v4 <= 0x40 && **(_QWORD **)a1 <= a2 )
      return **(_QWORD **)a1;
  }
  else
  {
    result = a2;
    if ( *(_QWORD *)a1 <= a2 )
      return *(_QWORD *)a1;
  }
  return result;
}
