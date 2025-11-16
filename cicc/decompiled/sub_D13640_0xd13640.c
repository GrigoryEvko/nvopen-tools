// Function: sub_D13640
// Address: 0xd13640
//
__int64 __fastcall sub_D13640(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned int v8; // eax
  char v9; // al

  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v3 != 30 || *(_BYTE *)(a1 + 24) )
  {
    v4 = *(_QWORD *)(a1 + 8);
    if ( v3 == v4 )
    {
      v9 = *(_BYTE *)(a1 + 25) ^ 1;
    }
    else
    {
      v5 = *(_QWORD *)(v3 + 40);
      v6 = *(_QWORD *)(a1 + 16);
      if ( v5 )
      {
        v7 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
        v8 = *(_DWORD *)(v5 + 44) + 1;
      }
      else
      {
        v7 = 0;
        v8 = 0;
      }
      if ( v8 >= *(_DWORD *)(v6 + 32) || !*(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v7) )
        return 2;
      v9 = sub_D0EBA0(v3, v4, 0, v6, *(_QWORD *)(a1 + 32)) ^ 1;
    }
    if ( !v9 )
    {
      *(_BYTE *)(a1 + 26) = 1;
      return 0;
    }
  }
  return 2;
}
