// Function: sub_250CBE0
// Address: 0x250cbe0
//
unsigned __int8 *__fastcall sub_250CBE0(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  int v3; // eax
  unsigned __int8 *result; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rax

  v2 = (unsigned __int8 *)(*a1 & 0xFFFFFFFFFFFFFFFCLL);
  if ( (*a1 & 3) == 3 )
    v2 = (unsigned __int8 *)*((_QWORD *)v2 + 3);
  v3 = *v2;
  if ( (unsigned __int8)v3 <= 0x1Cu )
    return (unsigned __int8 *)sub_25096F0(a1);
  v5 = (unsigned int)(v3 - 34);
  if ( (unsigned __int8)v5 > 0x33u )
    return (unsigned __int8 *)sub_25096F0(a1);
  v6 = 0x8000000000041LL;
  if ( !_bittest64(&v6, v5) )
    return (unsigned __int8 *)sub_25096F0(a1);
  v7 = sub_250C680(a1);
  if ( v7 )
    return *(unsigned __int8 **)(v7 + 24);
  result = sub_BD3990(*((unsigned __int8 **)v2 - 4), a2);
  if ( result )
  {
    if ( *result )
      return 0;
  }
  return result;
}
