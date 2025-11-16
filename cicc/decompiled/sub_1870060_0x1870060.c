// Function: sub_1870060
// Address: 0x1870060
//
char __fastcall sub_1870060(__int64 a1, __int64 a2)
{
  char result; // al
  unsigned __int8 v5; // dl
  __int64 v6; // rdi
  unsigned __int8 *v7; // rsi
  __int64 v8; // rbx
  size_t v9; // rdx
  int v10; // eax
  __int64 v11; // rdx

  result = sub_15E4F60(a2);
  if ( !result )
  {
    v5 = *(_BYTE *)(a2 + 32) & 0xF;
    if ( v5 == 1 || (*(_BYTE *)(a2 + 33) & 3) == 2 )
    {
      return 1;
    }
    else if ( (unsigned int)v5 - 7 > 1 )
    {
      v6 = a1 + 32;
      v7 = (unsigned __int8 *)sub_1649960(a2);
      v8 = *(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40);
      v10 = sub_16D1B30((__int64 *)(a1 + 32), v7, v9);
      v11 = v10 == -1 ? *(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40) : *(_QWORD *)(a1 + 32) + 8LL * v10;
      result = 1;
      if ( v8 == v11 )
      {
        if ( !*(_QWORD *)(a1 + 16) )
          sub_4263D6(v6, v7, v11);
        return (*(__int64 (__fastcall **)(__int64, __int64))(a1 + 24))(a1, a2);
      }
    }
  }
  return result;
}
