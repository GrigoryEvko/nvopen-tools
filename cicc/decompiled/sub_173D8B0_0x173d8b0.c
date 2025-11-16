// Function: sub_173D8B0
// Address: 0x173d8b0
//
char __fastcall sub_173D8B0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rdi
  void *v7; // r12
  char result; // al
  char v9; // dl
  char v10; // dl

  v6 = a1 + 8;
  v7 = sub_16982C0();
  if ( *(void **)(a1 + 8) == v7 )
    result = sub_16A0F40(v6, a2, a3, a4, a5);
  else
    result = sub_16984B0(v6);
  if ( result )
    return 0;
  if ( v7 == *(void **)(a1 + 8) )
  {
    v10 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 26LL) & 7;
    if ( v10 != 1 )
      return v10 != 0 && v10 != 3;
  }
  else
  {
    v9 = *(_BYTE *)(a1 + 26) & 7;
    if ( v9 != 1 )
      return v9 != 0 && v9 != 3;
  }
  return result;
}
