// Function: sub_2F5FCF0
// Address: 0x2f5fcf0
//
unsigned __int64 __fastcall sub_2F5FCF0(__int64 a1, signed int a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // esi
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 result; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  char v11; // dl
  bool v12; // si
  unsigned __int64 v13; // rdi
  bool v14; // dl

  if ( a2 >= 0 )
  {
    result = sub_2F5FA90(a1, a2, a4);
    if ( result )
      return (*(__int64 (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, result, 0);
  }
  else
  {
    v5 = a2 & 0x7FFFFFFF;
    v6 = *(_QWORD *)(*(_QWORD *)(a3 + 56) + 16LL * v5);
    if ( !v6 )
      return 0;
    v7 = (v6 >> 2) & 1;
    if ( ((v6 >> 2) & 1) != 0 )
      return v6 & 0xFFFFFFFFFFFFFFF8LL;
    if ( ((v6 >> 2) & 1) != 0 )
      return 0;
    v9 = v6 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v9 )
      return 0;
    if ( v5 >= *(_DWORD *)(a3 + 464) )
    {
      v12 = 0;
      v14 = 0;
      v13 = 0;
    }
    else
    {
      v10 = *(_QWORD *)(*(_QWORD *)(a3 + 456) + 8LL * v5);
      v11 = v10;
      LOBYTE(v7) = v10 & 1;
      v12 = (v10 & 4) != 0;
      v13 = v10 >> 3;
      v14 = (v11 & 2) != 0;
    }
    return (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64))(*(_QWORD *)a1 + 32LL))(
             a1,
             v9,
             (8 * v13) | (4LL * v12) | (unsigned __int8)v7 | (2LL * v14));
  }
  return result;
}
