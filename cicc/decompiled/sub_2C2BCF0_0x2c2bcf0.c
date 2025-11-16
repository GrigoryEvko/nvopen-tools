// Function: sub_2C2BCF0
// Address: 0x2c2bcf0
//
__int64 __fastcall sub_2C2BCF0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax

  v2 = sub_2BF04A0(a2);
  v3 = 0;
  if ( v2 )
  {
    if ( *(_BYTE *)(v2 + 8) == 4 && *(_BYTE *)(v2 + 160) == 70 )
    {
      v5 = **(_QWORD **)(v2 + 48);
      if ( v5 )
      {
        v3 = 1;
        **a1 = v5;
      }
    }
  }
  return v3;
}
