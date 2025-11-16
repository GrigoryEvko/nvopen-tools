// Function: sub_138E590
// Address: 0x138e590
//
__int64 *__fastcall sub_138E590(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax

  v2 = sub_138E440(a1, a2);
  if ( *((_BYTE *)v2 + 416) )
    return v2 + 8;
  else
    return 0;
}
