// Function: sub_7E1820
// Address: 0x7e1820
//
__int64 __fastcall sub_7E1820(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  __int64 v3; // rdx

  result = a1;
  if ( !a1 )
  {
    v2 = *(_QWORD *)(unk_4D03F68 + 16LL);
    if ( v2 )
    {
      while ( 1 )
      {
        result = *(_QWORD *)(v2 + 40);
        if ( unk_4D044B4 && *(_BYTE *)v2 == 2 )
        {
          while ( result )
          {
            if ( (*(_BYTE *)(result + 49) & 0x10) == 0 )
              goto LABEL_12;
            result = *(_QWORD *)(result + 32);
          }
        }
        else
        {
          while ( result )
          {
LABEL_12:
            v3 = *(_QWORD *)(result + 80);
            if ( !v3 || ((*(_BYTE *)(result + 50) & 4) == 0 || *(_BYTE *)(v3 + 128)) && !*(_BYTE *)(v3 + 132) )
              return result;
            result = *(_QWORD *)(result + 32);
          }
        }
        v2 = *(_QWORD *)(v2 + 32);
        if ( !v2 )
          return 0;
      }
    }
  }
  return result;
}
