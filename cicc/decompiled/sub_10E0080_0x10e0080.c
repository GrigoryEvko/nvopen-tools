// Function: sub_10E0080
// Address: 0x10e0080
//
unsigned __int64 __fastcall sub_10E0080(_DWORD *a1, unsigned __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int64 result; // rax
  unsigned int v4; // r13d

  v2 = a1[2];
  if ( v2 > 0x40 )
  {
    v4 = v2 - sub_C444A0((__int64)a1);
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
