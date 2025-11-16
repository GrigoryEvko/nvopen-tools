// Function: sub_8D4A00
// Address: 0x8d4a00
//
unsigned __int64 __fastcall sub_8D4A00(__int64 a1)
{
  char v1; // al
  int v2; // edx
  unsigned int v3; // eax
  unsigned __int64 result; // rax
  unsigned __int64 i; // rdx

  v1 = *(_BYTE *)(a1 + 140);
  if ( v1 != 12 )
    goto LABEL_14;
  v2 = 0;
  do
  {
    v3 = *(unsigned __int8 *)(a1 + 185);
    a1 = *(_QWORD *)(a1 + 160);
    v2 |= (v3 >> 3) & 1;
    v1 = *(_BYTE *)(a1 + 140);
  }
  while ( v1 == 12 );
  if ( v2 && (_DWORD)qword_4F077B4 )
  {
    result = *(_QWORD *)(a1 + 128);
    if ( result <= unk_4F06AD8 && result > 2 )
    {
      if ( result <= 4 )
      {
        return 4;
      }
      else
      {
        for ( i = 4; i < result; i *= 2LL )
          ;
        return i;
      }
    }
  }
  else
  {
LABEL_14:
    if ( dword_4F077C0 && (v1 == 1 || v1 == 7) )
      return 1;
    else
      return *(_QWORD *)(a1 + 128);
  }
  return result;
}
