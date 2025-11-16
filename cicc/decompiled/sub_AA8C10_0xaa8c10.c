// Function: sub_AA8C10
// Address: 0xaa8c10
//
__int64 __fastcall sub_AA8C10(__int64 a1)
{
  _QWORD *v1; // rbx
  _BYTE *v3; // rdi
  unsigned int v4; // r13d

  v1 = (_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  if ( (_QWORD *)a1 == v1 )
    return 1;
  while ( 1 )
  {
    v3 = (_BYTE *)*v1;
    if ( *(_BYTE *)*v1 != 17 )
      break;
    v4 = *((_DWORD *)v3 + 8);
    if ( v4 <= 0x40 )
    {
      if ( *((_QWORD *)v3 + 3) )
        return 0;
    }
    else if ( v4 != (unsigned int)sub_C444A0(v3 + 24) )
    {
      return 0;
    }
    v1 += 4;
    if ( (_QWORD *)a1 == v1 )
      return 1;
  }
  return 0;
}
