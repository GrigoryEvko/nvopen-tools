// Function: sub_34B85F0
// Address: 0x34b85f0
//
bool __fastcall sub_34B85F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool result; // al
  int v6; // eax
  unsigned __int16 v8; // ax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int16 v12; // ax

  result = 1;
  if ( a1 != a2 )
  {
    v6 = *(unsigned __int8 *)(a1 + 8);
    if ( (_BYTE)v6 == 14 )
      return *(_BYTE *)(a2 + 8) == 14;
    else
      return (unsigned int)(v6 - 17) <= 1
          && (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1
          && (v8 = sub_30097B0(a1, 0, a3, a4, a5)) != 0
          && *(_QWORD *)(a3 + 8LL * v8 + 112)
          && (v12 = sub_30097B0(a2, 0, v9, v10, v11)) != 0
          && *(_QWORD *)(a3 + 8LL * v12 + 112) != 0;
  }
  return result;
}
