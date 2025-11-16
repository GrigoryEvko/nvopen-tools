// Function: sub_160CB00
// Address: 0x160cb00
//
__int64 __fastcall sub_160CB00(__int64 a1, __int64 a2)
{
  unsigned int v3; // ebx
  unsigned int v4; // r13d
  __int64 v5; // rdx
  __int64 v6; // rdi

  if ( !*(_DWORD *)(a1 + 192) )
    return 0;
  v3 = 0;
  v4 = 0;
  do
  {
    v5 = v3++;
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v5);
    v4 |= (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 24LL))(v6, a2);
  }
  while ( *(_DWORD *)(a1 + 192) > v3 );
  return v4;
}
