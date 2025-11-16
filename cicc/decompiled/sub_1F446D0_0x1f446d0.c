// Function: sub_1F446D0
// Address: 0x1f446d0
//
__int64 __fastcall sub_1F446D0(void *a1, __int64 a2)
{
  __int64 v2; // r8
  bool v3; // zf
  bool v5; // zf

  if ( &unk_4FC8CB4 == a1 )
  {
    v3 = byte_4FCDEA0 == 0;
    goto LABEL_19;
  }
  if ( &unk_4FCF234 == a1 )
  {
    v3 = byte_4FCDDC0 == 0;
    goto LABEL_19;
  }
  if ( &unk_4FCAE51 == a1 )
  {
    v3 = byte_4FCDCE0 == 0;
    goto LABEL_19;
  }
  if ( &unk_4FCAE50 == a1 )
  {
    v3 = byte_4FCDC00 == 0;
    goto LABEL_19;
  }
  if ( &unk_4FC4B0C == a1 )
  {
    v3 = byte_4FCDB20 == 0;
    goto LABEL_19;
  }
  if ( &unk_4FCAC8C == a1 )
  {
    v3 = byte_4FCD960 == 0;
    goto LABEL_19;
  }
  if ( &unk_4FC332C == a1 )
  {
    v3 = byte_4FCD880 == 0;
LABEL_19:
    v2 = 0;
    if ( v3 )
      return a2;
    return v2;
  }
  if ( &unk_4FC3344 == a1 )
  {
    v5 = byte_4FCD7A0 == 0;
  }
  else if ( &unk_4FC64C8 == a1 )
  {
    v5 = byte_4FCD6C0 == 0;
  }
  else
  {
    if ( &unk_4FC5C94 != a1 )
    {
      if ( &unk_4FC64C9 == a1 )
      {
        v3 = byte_4FCD420 == 0;
      }
      else if ( &unk_4FC7F74 == a1 )
      {
        v3 = byte_4FCD340 == 0;
      }
      else
      {
        if ( &unk_4FC7F6C != a1 )
        {
          v2 = a2;
          if ( &unk_4FC5C8C == a1 && byte_4FCCEE0 )
            return 0;
          return v2;
        }
        v3 = byte_4FCD260 == 0;
      }
      goto LABEL_19;
    }
    v5 = byte_4FCD5E0 == 0;
  }
  v2 = 0;
  if ( v5 )
    return a2;
  return v2;
}
