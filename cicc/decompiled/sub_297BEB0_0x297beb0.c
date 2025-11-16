// Function: sub_297BEB0
// Address: 0x297beb0
//
char __fastcall sub_297BEB0(__int64 a1, __int64 a2, __int64 *a3)
{
  char result; // al
  unsigned int v5; // edx
  unsigned __int64 v6; // rbx
  __int64 v7; // rax

  result = 0;
  v5 = *(_DWORD *)(a2 + 32);
  if ( v5 <= 0x40 )
  {
    v6 = 0;
    if ( v5 )
      v6 = (__int64)(*(_QWORD *)(a2 + 24) << (64 - (unsigned __int8)v5)) >> (64 - (unsigned __int8)v5);
    v7 = sub_D95540(a1);
    return sub_DFA150(a3, v7, 0, 0, 1u, v6);
  }
  return result;
}
