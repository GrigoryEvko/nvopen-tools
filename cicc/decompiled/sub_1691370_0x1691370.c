// Function: sub_1691370
// Address: 0x1691370
//
const char *__fastcall sub_1691370(__int64 *a1, int a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // r8

  while ( 1 )
  {
    v2 = *a1;
    if ( !*(_WORD *)(*a1 + ((unsigned __int64)(unsigned int)(a2 - 1) << 6) + 40) )
      break;
    v3 = v2
       + ((unsigned __int64)((unsigned int)*(unsigned __int16 *)(*a1
                                                               + ((unsigned __int64)(unsigned int)(a2 - 1) << 6)
                                                               + 40)
                           - 1) << 6);
    v4 = *(_QWORD *)(v3 + 16);
    if ( v4 )
      return (const char *)v4;
    a2 = *(unsigned __int16 *)(v3 + 40);
    if ( !*(_WORD *)(v3 + 40) )
      break;
    v4 = *(_QWORD *)(v2 + ((unsigned __int64)(unsigned int)(a2 - 1) << 6) + 16);
    if ( v4 )
      return (const char *)v4;
  }
  return "OPTIONS";
}
