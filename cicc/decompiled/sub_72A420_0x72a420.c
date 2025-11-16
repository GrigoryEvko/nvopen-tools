// Function: sub_72A420
// Address: 0x72a420
//
__int64 __fastcall sub_72A420(__int64 *a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  __int64 v4; // rax

  result = *((unsigned __int8 *)a1 + 169);
  if ( (result & 0x40) == 0 )
  {
    *((_BYTE *)a1 + 169) = result | 0x40;
    v3 = *a1;
    if ( v3 )
    {
      if ( dword_4F04C44 == -1 )
      {
        v4 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v4 + 6) & 2) == 0 && (dword_4F04C64 == -1 || (*(_BYTE *)(v4 + 14) & 2) == 0) )
          sub_8AD0D0(v3, 1, 1);
      }
    }
    if ( (a1[21] & 0x80008000) != 0 )
      *((_BYTE *)a1 + 171) |= 0x10u;
    result = dword_4D03F94;
    if ( dword_4D03F94 )
    {
      if ( *((_BYTE *)a1 + 136) == 5 )
        *((_BYTE *)a1 + 136) = 3;
    }
  }
  return result;
}
