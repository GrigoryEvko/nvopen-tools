// Function: sub_2F7E330
// Address: 0x2f7e330
//
__int64 __fastcall sub_2F7E330(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // dl
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // r14d
  int v7; // ebx
  int v8; // eax

  if ( !sub_B92180(*a2) )
    return 0;
  v2 = sub_B92180(*a2);
  v3 = *(_BYTE *)(v2 - 16);
  v4 = (v3 & 2) != 0 ? *(_QWORD *)(v2 - 32) : v2 - 16 - 8LL * ((v3 >> 2) & 0xF);
  if ( !*(_DWORD *)(*(_QWORD *)(v4 + 40) + 32LL) )
    return 0;
  v5 = a2[41];
  if ( (__int64 *)v5 == a2 + 40 )
    return 0;
  v6 = 0;
  do
  {
    v7 = sub_2F7D880(v5);
    v8 = sub_2F7D060(v5);
    v5 = *(_QWORD *)(v5 + 8);
    v6 |= v8 | v7;
  }
  while ( a2 + 40 != (__int64 *)v5 );
  return v6;
}
