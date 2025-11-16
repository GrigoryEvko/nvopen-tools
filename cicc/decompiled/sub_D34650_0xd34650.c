// Function: sub_D34650
// Address: 0xd34650
//
char __fastcall sub_D34650(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  char result; // al

  v3 = *(_QWORD *)(a1 + 8);
  v4 = v3 + 72LL * a2;
  v5 = v3 + 72LL * a3;
  if ( *(_BYTE *)(v4 + 40) || (result = *(_BYTE *)(v5 + 40)) != 0 )
  {
    result = 0;
    if ( *(_DWORD *)(v4 + 44) != *(_DWORD *)(v5 + 44) )
      return *(_DWORD *)(v4 + 48) == *(_DWORD *)(v5 + 48);
  }
  return result;
}
