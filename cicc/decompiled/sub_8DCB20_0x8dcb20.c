// Function: sub_8DCB20
// Address: 0x8dcb20
//
__int64 **__fastcall sub_8DCB20(__int64 a1)
{
  __int64 **result; // rax
  __int64 *i; // rbx
  __int64 v3; // r12
  unsigned int v4; // edx
  __int64 *v5; // rcx
  int v6; // eax
  char v7; // dl

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  result = *(__int64 ***)(a1 + 168);
  for ( i = *result; i; i = v5 )
  {
    while ( 1 )
    {
      v3 = i[1];
      if ( sub_8D3410(v3) )
        v3 = sub_8D67C0(v3);
      v4 = (unsigned int)sub_8DC000(v3) << 7;
      result = (__int64 **)(v4 | i[4] & 0x7F);
      *((_BYTE *)i + 32) = v4 | i[4] & 0x7F;
      if ( (char)result < 0 )
        break;
      i = (__int64 *)*i;
      if ( !i )
        return result;
    }
    if ( (*((_BYTE *)i + 33) & 1) == 0 || (v5 = (__int64 *)*i, v6 = 0, !*i) )
    {
      v6 = sub_8D9650(v3);
      v5 = (__int64 *)*i;
      LOBYTE(v6) = v6 != 0;
    }
    v7 = (_BYTE)v6 << 6;
    result = (__int64 **)((v6 << 6) | (_BYTE)i[4] & 0xBFu);
    *((_BYTE *)i + 32) = v7 | i[4] & 0xBF;
  }
  return result;
}
