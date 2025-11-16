// Function: sub_21546A0
// Address: 0x21546a0
//
_BYTE *__fastcall sub_21546A0(__int64 a1, _BYTE *a2)
{
  _BYTE *v2; // rcx
  _BYTE *v3; // rax
  int v4; // edx

  v2 = *(_BYTE **)(*(_QWORD *)(a1 + 56) + 16LL);
  if ( a2 < v2 )
  {
    v3 = a2;
    v4 = 0;
    while ( *v3 != 10 )
    {
      ++v3;
      ++v4;
      if ( v3 == v2 )
        goto LABEL_7;
    }
    if ( v3 < v2 )
      return a2;
  }
LABEL_7:
  *(_BYTE *)(a1 + 48) = 1;
  return a2;
}
