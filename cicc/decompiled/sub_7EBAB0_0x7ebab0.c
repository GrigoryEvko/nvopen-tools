// Function: sub_7EBAB0
// Address: 0x7ebab0
//
__int64 __fastcall sub_7EBAB0(__int64 a1, __m128i **a2)
{
  char v2; // r13
  __m128i *v3; // rax
  unsigned int v4; // r8d

  v2 = *(_BYTE *)(a1 + 173);
  if ( v2 != 10 )
  {
    if ( v2 == 2 )
    {
      if ( (*(_BYTE *)(a1 + 170) & 0x40) == 0 )
      {
        v3 = 0;
        v4 = 0;
        goto LABEL_4;
      }
    }
    else
    {
      v3 = 0;
      v4 = 0;
      if ( v2 != 7 || (*(_BYTE *)(a1 + 192) & 2) == 0 )
        goto LABEL_4;
    }
  }
  if ( (unsigned int)sub_7E1F90(*(_QWORD *)(a1 + 128)) || (*(_BYTE *)(a1 + 168) & 0x20) != 0 )
  {
    v3 = sub_7EB890(a1, 1);
    v4 = 1;
  }
  else
  {
    v3 = sub_7EB890(a1, v2 != 10);
    v4 = 1;
  }
LABEL_4:
  *a2 = v3;
  return v4;
}
