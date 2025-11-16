// Function: sub_12A1990
// Address: 0x12a1990
//
__int64 __fastcall sub_12A1990(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  char v3; // cl
  int v4; // esi
  __int64 v5; // rax
  char v6; // si
  int v7; // r13d
  __int64 v8; // rbx
  __int64 v9; // rsi
  _DWORD v11[11]; // [rsp+4h] [rbp-2Ch]

  v2 = sub_12A0C10(a1, *(_QWORD *)(a2 + 160));
  v3 = *(_BYTE *)(a2 + 185) & 0x7F;
  if ( (*(_BYTE *)(a2 + 185) & 2) != 0 )
  {
    v4 = 53;
    v5 = 1;
  }
  else
  {
    v4 = v11[0];
    v5 = 0;
  }
  v11[0] = v4;
  v6 = v3 & 4;
  if ( (v3 & 1) != 0 )
  {
    v11[v5] = 38;
    v7 = v5 + 1;
    if ( v6 )
    {
      v11[v7] = 55;
      v7 = v5 + 2;
    }
  }
  else if ( v6 )
  {
    v11[v5] = 55;
    v7 = v5 + 1;
  }
  else
  {
    if ( !(_DWORD)v5 )
      return v2;
    v7 = v5;
  }
  v8 = 0;
  do
  {
    v9 = (unsigned int)v11[v8++];
    v2 = sub_15A5A60(a1 + 16, v9, v2);
  }
  while ( v7 > (int)v8 );
  return v2;
}
