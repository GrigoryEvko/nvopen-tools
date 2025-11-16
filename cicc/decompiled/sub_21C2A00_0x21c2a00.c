// Function: sub_21C2A00
// Address: 0x21c2a00
//
bool __fastcall sub_21C2A00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r10d
  bool result; // al
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // rax

  while ( 1 )
  {
    v4 = *(unsigned __int16 *)(a2 + 24);
    result = (_WORD)v4 == 39 || (_WORD)v4 == 34;
    if ( result )
    {
      *(_QWORD *)a4 = a2;
      *(_DWORD *)(a4 + 8) = a3;
      return result;
    }
    if ( v4 == 260 )
      break;
    if ( (_WORD)v4 != 159 )
      return result;
    if ( *(_DWORD *)(a2 + 84) )
      return result;
    if ( *(_DWORD *)(a2 + 88) != 101 )
      return result;
    v7 = **(_QWORD **)(a2 + 32);
    if ( *(_WORD *)(v7 + 24) != 287 )
      return result;
    v8 = *(__int64 **)(v7 + 32);
    a2 = *v8;
    a3 = v8[1];
  }
  v6 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)a4 = *(_QWORD *)v6;
  *(_DWORD *)(a4 + 8) = *(_DWORD *)(v6 + 8);
  return 1;
}
