// Function: sub_87E350
// Address: 0x87e350
//
__int64 __fastcall sub_87E350(__int64 a1)
{
  __int64 *v1; // rbx
  unsigned __int8 v2; // r12
  __int64 result; // rax

  v1 = *(__int64 **)(a1 + 32);
  if ( v1 )
  {
    if ( (*(_BYTE *)(a1 + 64) & 4) != 0 )
    {
      v2 = 8;
      do
      {
LABEL_4:
        result = sub_685A50(*((unsigned int *)v1 + 4), (_DWORD *)v1 + 2, (FILE *)v1[3], v2);
        v1 = (__int64 *)*v1;
      }
      while ( v1 );
      return result;
    }
    result = (unsigned int)dword_4D04964;
    if ( dword_4D04964 )
    {
      result = (__int64)&unk_4F07471;
      v2 = unk_4F07471;
      if ( unk_4F07471 != 3 )
        goto LABEL_4;
    }
  }
  return result;
}
