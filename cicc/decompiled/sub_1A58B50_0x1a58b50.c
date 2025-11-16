// Function: sub_1A58B50
// Address: 0x1a58b50
//
void *__fastcall sub_1A58B50(_BYTE *src, _BYTE *a2, __int64 a3)
{
  void *v3; // rdx

  v3 = (void *)(a3 - (a2 - src));
  if ( a2 == src )
    return v3;
  else
    return memmove(v3, src, a2 - src);
}
