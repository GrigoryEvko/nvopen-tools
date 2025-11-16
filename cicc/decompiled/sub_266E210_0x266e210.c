// Function: sub_266E210
// Address: 0x266e210
//
__int64 __fastcall sub_266E210(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax

  v2 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)v2 != 85 || a1 != v2 - 32 )
    return 0;
  if ( *(char *)(v2 + 7) < 0 )
  {
    v4 = sub_BD2BC0(*(_QWORD *)(a1 + 24));
    v6 = v4 + v5;
    v7 = 0;
    if ( *(char *)(v2 + 7) < 0 )
      v7 = sub_BD2BC0(v2);
    if ( (unsigned int)((v6 - v7) >> 4) )
      return 0;
  }
  if ( a2 )
  {
    v8 = *(_QWORD *)(a2 + 120);
    if ( !v8 )
      return 0;
    v9 = *(_QWORD *)(v2 - 32);
    if ( !v9 || *(_BYTE *)v9 || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v2 + 80) || v8 != v9 )
      return 0;
  }
  return v2;
}
