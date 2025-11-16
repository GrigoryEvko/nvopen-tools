// Function: sub_73BC00
// Address: 0x73bc00
//
__int64 **__fastcall sub_73BC00(__int64 a1, __int64 a2)
{
  __int64 **i; // rbx
  __int64 **result; // rax
  __int64 *j; // r12
  int v5; // eax
  __int64 v6; // rdi

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  for ( i = **(__int64 ****)(a1 + 168); *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  result = *(__int64 ***)(a2 + 168);
  for ( j = *result; i; j = (__int64 *)*j )
  {
    if ( ((_BYTE)i[4] & 4) != 0 )
    {
      v5 = *((unsigned __int8 *)j + 32) | 4;
      *((_BYTE *)j + 32) = v5;
      result = (__int64 **)((_BYTE)i[4] & 8 | v5 & 0xFFFFFFF7);
      *((_BYTE *)j + 32) = (_BYTE)result;
      if ( ((_BYTE)i[4] & 0x10) != 0 )
      {
        result = (__int64 **)((unsigned int)result | 0x10);
        j[6] = (__int64)i;
        *((_BYTE *)j + 32) = (_BYTE)result;
      }
      v6 = (__int64)i[5];
      if ( v6 )
      {
        if ( !j[5] )
        {
          result = (__int64 **)sub_73BB50(v6);
          j[5] = (__int64)result;
        }
      }
    }
    i = (__int64 **)*i;
  }
  return result;
}
