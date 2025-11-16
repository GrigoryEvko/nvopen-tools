// Function: sub_385DBB0
// Address: 0x385dbb0
//
char __fastcall sub_385DBB0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rdx
  char result; // al

  v3 = *(_QWORD *)(a1 + 8);
  v4 = v3 + ((unsigned __int64)a2 << 6);
  v5 = v3 + ((unsigned __int64)a3 << 6);
  if ( *(_BYTE *)(v4 + 40) || (result = *(_BYTE *)(v5 + 40)) != 0 )
  {
    result = 0;
    if ( *(_DWORD *)(v4 + 44) != *(_DWORD *)(v5 + 44) )
      return *(_DWORD *)(v4 + 48) == *(_DWORD *)(v5 + 48);
  }
  return result;
}
