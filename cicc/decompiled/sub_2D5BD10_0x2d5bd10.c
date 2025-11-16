// Function: sub_2D5BD10
// Address: 0x2d5bd10
//
bool __fastcall sub_2D5BD10(__int64 a1, __int64 a2, __int64 **a3)
{
  int v4; // esi
  unsigned int v6; // eax
  __int64 v7; // r13
  unsigned __int16 v9; // ax
  char v10; // al

  v4 = *(unsigned __int8 *)a3;
  if ( (unsigned __int8)v4 <= 0x1Cu )
    return 0;
  v6 = sub_2FEBEF0(a1, (unsigned int)(v4 - 29));
  v7 = v6;
  if ( !v6 )
    return 1;
  v9 = sub_2D5BAE0(a1, a2, a3[1], 0);
  if ( v9 != 1 && (!v9 || !*(_QWORD *)(a1 + 8LL * v9 + 112)) )
    return 0;
  if ( (unsigned int)v7 > 0x1F3 )
    return 1;
  v10 = *(_BYTE *)(v7 + 500LL * v9 + a1 + 6414);
  if ( !v10 )
    return 1;
  return v10 == 4;
}
