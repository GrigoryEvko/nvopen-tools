// Function: sub_730E00
// Address: 0x730e00
//
__int64 __fastcall sub_730E00(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // rdx
  char v3; // cl
  __int64 v4; // rdx

  if ( *(_BYTE *)(a1 + 173) != 12 || *(_BYTE *)(a1 + 176) != 1 )
    return a1;
  v2 = sub_72E9A0(a1);
  result = a1;
  if ( *((_BYTE *)v2 + 24) == 1 )
  {
    v3 = *((_BYTE *)v2 + 56);
    if ( v3 == 5 || v3 == 116 )
    {
      result = a1;
      if ( (*((_BYTE *)v2 + 27) & 2) != 0 )
      {
        v4 = v2[9];
        if ( *(_BYTE *)(v4 + 24) == 2 )
          result = *(_QWORD *)(v4 + 56);
      }
    }
  }
  if ( *(_BYTE *)(result + 173) != 12 )
    return a1;
  return result;
}
