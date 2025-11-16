// Function: sub_2D05370
// Address: 0x2d05370
//
__int64 __fastcall sub_2D05370(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rdi
  char v4; // r9
  __int64 v5; // r8
  __int64 v6; // r10

  v3 = *(_QWORD *)(a2 + 24);
  v4 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 <= 0x1Cu )
    return 0;
  v5 = a2;
  if ( a3 && !sub_2D04640(v3) )
    return v6;
  if ( v4 == 84 )
    return *(_QWORD *)(*(_QWORD *)(v3 - 8)
                     + 32LL * *(unsigned int *)(v3 + 72)
                     + 8LL * (unsigned int)((v5 - *(_QWORD *)(v3 - 8)) >> 5));
  return *(_QWORD *)(v3 + 40);
}
