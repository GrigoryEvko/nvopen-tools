// Function: sub_10DF440
// Address: 0x10df440
//
bool __fastcall sub_10DF440(unsigned int *a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int64 v3; // r13
  _QWORD *v4; // rax
  bool result; // al
  unsigned int v6; // r12d

  v2 = *(_DWORD *)(a2 + 8);
  v3 = *a1;
  if ( v2 <= 0x40 )
  {
    v4 = *(_QWORD **)a2;
    return v3 > (unsigned __int64)v4;
  }
  v6 = v2 - sub_C444A0(a2);
  result = 0;
  if ( v6 <= 0x40 )
  {
    v4 = **(_QWORD ***)a2;
    return v3 > (unsigned __int64)v4;
  }
  return result;
}
