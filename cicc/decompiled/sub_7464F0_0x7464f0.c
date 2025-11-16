// Function: sub_7464F0
// Address: 0x7464f0
//
char *__fastcall sub_7464F0(__int64 a1, int a2)
{
  char v2; // al
  char *v4; // r8

  v2 = *(_BYTE *)(a1 + 160);
  if ( !(unk_4D04548 | unk_4D04558) || (*(_BYTE *)(a1 + 161) & 2) == 0 )
    return sub_7462A0(v2, a2);
  v4 = "__int8";
  if ( unk_4F06AD1 != v2 )
  {
    v4 = "unsigned __int8";
    if ( unk_4F06AD0 != v2 )
    {
      v4 = "__int16";
      if ( unk_4F06ACF != v2 )
      {
        v4 = "unsigned __int16";
        if ( unk_4F06ACE != v2 )
        {
          v4 = "__int32";
          if ( unk_4F06ACD != v2 )
          {
            v4 = "unsigned __int32";
            if ( unk_4F06ACC != v2 )
            {
              v4 = "__int64";
              if ( unk_4F06ACB != v2 )
              {
                v4 = "unsigned __int64";
                if ( unk_4F06ACA != v2 )
                  return "**BAD-SIZED-INT-KIND**";
              }
            }
          }
        }
      }
    }
  }
  return v4;
}
