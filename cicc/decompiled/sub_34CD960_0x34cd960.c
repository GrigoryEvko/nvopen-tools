// Function: sub_34CD960
// Address: 0x34cd960
//
bool __fastcall sub_34CD960(__int64 a1)
{
  _DWORD *v1; // rbx
  bool result; // al
  unsigned int v3; // ecx

  v1 = *(_DWORD **)(*(_QWORD *)(a1 + 32) + 8LL);
  if ( !sub_23CF1B0((__int64)v1) )
    return 0;
  if ( (unsigned int)(v1[159] - 3) <= 1 )
    return 0;
  result = sub_CC7F40((__int64)(v1 + 128));
  if ( !result )
    return 0;
  if ( v1[136] == 3 )
  {
    v3 = v1[139];
    if ( v3 <= 0x1F )
      return ((0xD8000222uLL >> v3) & 1) == 0;
  }
  return result;
}
