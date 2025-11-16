// Function: sub_C33940
// Address: 0xc33940
//
bool __fastcall sub_C33940(__int64 a1)
{
  bool result; // al
  int v2; // r12d
  __int64 v3; // rax

  result = (*(_BYTE *)(a1 + 20) & 6) != 0 && (*(_BYTE *)(a1 + 20) & 7) != 3;
  if ( result )
  {
    result = 0;
    if ( *(_DWORD *)(a1 + 16) == *(_DWORD *)(*(_QWORD *)a1 + 4LL) )
    {
      v2 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
      v3 = sub_C33930(a1);
      return (unsigned int)sub_C45D90(v3, (unsigned int)(v2 - 1)) == 0;
    }
  }
  return result;
}
