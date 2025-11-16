// Function: sub_B8F9E0
// Address: 0xb8f9e0
//
char __fastcall sub_B8F9E0(__int64 a1, __int64 a2)
{
  char result; // al
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 *v5; // rsi
  unsigned int v6; // edx
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rcx

  if ( a1 == a2 )
    return 1;
  result = 0;
  if ( a1 && *(_BYTE *)a1 == 1 && a2 && *(_BYTE *)a2 == 1 )
  {
    v3 = *(_QWORD *)(a1 + 136);
    v4 = *(_QWORD *)(a2 + 136);
    v5 = *(__int64 **)(v3 + 24);
    v6 = *(_DWORD *)(v3 + 32);
    if ( v6 > 0x40 )
    {
      v7 = *v5;
    }
    else
    {
      v7 = 0;
      if ( v6 )
        v7 = (__int64)((_QWORD)v5 << (64 - (unsigned __int8)v6)) >> (64 - (unsigned __int8)v6);
    }
    v8 = *(_QWORD **)(v4 + 24);
    v9 = *(_DWORD *)(v4 + 32);
    if ( v9 > 0x40 )
    {
      return *v8 == v7;
    }
    else
    {
      v10 = 0;
      if ( v9 )
        v10 = (__int64)((_QWORD)v8 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
      return v10 == v7;
    }
  }
  return result;
}
