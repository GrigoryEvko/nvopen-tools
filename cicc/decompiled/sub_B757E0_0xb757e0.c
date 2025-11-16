// Function: sub_B757E0
// Address: 0xb757e0
//
unsigned __int64 __fastcall sub_B757E0(__int64 a1, unsigned int a2)
{
  unsigned __int8 v2; // al
  __int64 v4; // rax

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
    return sub_B751D0(
             (__int64 *)(*(_QWORD *)(a1 - 32) + 8LL * a2),
             (__int64 *)(*(_QWORD *)(a1 - 32) + 8LL * *(unsigned int *)(a1 - 24)));
  v4 = a1 - 8LL * ((v2 >> 2) & 0xF) - 16;
  return sub_B751D0((__int64 *)(v4 + 8LL * a2), (__int64 *)(v4 + 8LL * ((*(_WORD *)(a1 - 16) >> 6) & 0xF)));
}
