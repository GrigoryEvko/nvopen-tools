// Function: sub_15AFB30
// Address: 0x15afb30
//
__int64 __fastcall sub_15AFB30(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // r8d
  unsigned int v4; // ecx
  __int64 v6; // rax
  char v7; // al
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = 0;
  v4 = *(_DWORD *)(a2 + 8);
  if ( *(_QWORD *)a1 == *(_QWORD *)(a2 + 8 * (1 - v2))
    && *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8 * (2 - v2))
    && *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 + 8 * (3 - v2)) )
  {
    v6 = a2;
    if ( *(_BYTE *)a2 != 15 )
      v6 = *(_QWORD *)(a2 - 8 * v2);
    v3 = 0;
    if ( *(_QWORD *)(a1 + 24) == v6
      && *(_DWORD *)(a1 + 32) == *(_DWORD *)(a2 + 24)
      && *(_QWORD *)(a1 + 40) == *(_QWORD *)(a2 + 8 * (4 - v2)) )
    {
      v7 = *(_BYTE *)(a2 + 40);
      if ( *(_BYTE *)(a1 + 48) == ((v7 & 4) != 0)
        && *(_BYTE *)(a1 + 49) == ((v7 & 8) != 0)
        && *(_DWORD *)(a1 + 52) == *(_DWORD *)(a2 + 28) )
      {
        v8 = v4 <= 8 ? 0LL : *(_QWORD *)(a2 + 8 * (8 - v2));
        v3 = 0;
        if ( *(_QWORD *)(a1 + 56) == v8
          && *(_DWORD *)(a1 + 64) == (v7 & 3)
          && *(_DWORD *)(a1 + 68) == *(_DWORD *)(a2 + 32)
          && *(_DWORD *)(a1 + 72) == *(_DWORD *)(a2 + 36)
          && *(_DWORD *)(a1 + 76) == *(_DWORD *)(a2 + 44)
          && *(_BYTE *)(a1 + 80) == ((v7 & 0x10) != 0)
          && *(_QWORD *)(a1 + 88) == *(_QWORD *)(a2 + 8 * (5 - v2)) )
        {
          v9 = v4 <= 9 ? 0LL : *(_QWORD *)(a2 + 8 * (9 - v2));
          v3 = 0;
          if ( *(_QWORD *)(a1 + 96) == v9
            && *(_QWORD *)(a1 + 104) == *(_QWORD *)(a2 + 8 * (6 - v2))
            && *(_QWORD *)(a1 + 112) == *(_QWORD *)(a2 + 8 * (7 - v2)) )
          {
            v10 = *(_QWORD *)(a1 + 120);
            if ( v4 <= 0xA )
              v11 = 0;
            else
              v11 = *(_QWORD *)(a2 + 8 * (10 - v2));
            LOBYTE(v3) = v10 == v11;
          }
        }
      }
    }
  }
  return v3;
}
