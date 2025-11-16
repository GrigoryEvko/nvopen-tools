// Function: sub_8CC0D0
// Address: 0x8cc0d0
//
__int64 ***__fastcall sub_8CC0D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 ***result; // rax
  char v5; // r15
  __int64 **v6; // rbx
  __int64 **i; // r12

  v2 = *(_QWORD *)(a1 + 152);
  v3 = *(_QWORD *)(a2 + 152);
  result = (__int64 ***)sub_8CBB20(0xBu, a1, (_QWORD *)a2);
  if ( *(char *)(a1 + 192) < 0 && *(char *)(a2 + 192) < 0
    || (*(_BYTE *)(a1 + 195) & 1) != 0 && (*(_BYTE *)(a2 + 195) & 1) != 0 )
  {
    v5 = 1;
    if ( *(_BYTE *)(v2 + 140) != 7 )
      return result;
  }
  else
  {
    v5 = 0;
    if ( (*(_BYTE *)(a1 + 89) & 4) == 0 || *(_BYTE *)(v2 + 140) != 7 )
      return result;
  }
  if ( *(_BYTE *)(v3 + 140) == 7 )
  {
    v6 = **(__int64 ****)(v2 + 168);
    result = *(__int64 ****)(v3 + 168);
    for ( i = *result; v6; i = (__int64 **)*i )
    {
      if ( !i )
        break;
      if ( v5 || ((_BYTE)v6[4] & 8) != 0 || ((_BYTE)i[4] & 8) != 0 )
        result = (__int64 ***)sub_8CA8D0(v6[7], i[7]);
      v6 = (__int64 **)*v6;
    }
  }
  return result;
}
