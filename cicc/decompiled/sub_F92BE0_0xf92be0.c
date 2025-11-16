// Function: sub_F92BE0
// Address: 0xf92be0
//
__int64 __fastcall sub_F92BE0(__int64 a1, __int64 a2)
{
  int v3; // r13d
  unsigned int v4; // r8d
  _QWORD *v6; // r9
  _QWORD *v7; // rdi
  __int64 v8; // r9
  char *v9; // rax
  char *v10; // rdx
  char *v11; // rsi
  _QWORD *v12; // rcx
  _QWORD *v13; // rdx
  __int64 v14; // r10

  if ( (unsigned __int8)(*(_BYTE *)a1 - 82) > 1u || (unsigned __int8)(*(_BYTE *)a2 - 82) > 1u )
  {
    if ( sub_B46D50((unsigned __int8 *)a1) && (unsigned __int8)sub_B46250(a1, a2, 0) )
    {
      v6 = (*(_BYTE *)(a1 + 7) & 0x40) != 0
         ? *(_QWORD **)(a1 - 8)
         : (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v7 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
         ? *(_QWORD **)(a2 - 8)
         : (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      if ( *v6 == v7[4] && v6[4] == *v7 )
      {
        sub_F8F2E0((__int64)v7, (__int64)&v7[4 * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)], 2);
        v9 = (char *)sub_F8F2E0(v8, v8 + 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF), 2);
        v11 = v10;
        v13 = v12;
        if ( v11 - v9 == v14 - (_QWORD)v12 )
        {
          if ( v9 == v11 )
            return v4;
          while ( *(_QWORD *)v9 == *v13 )
          {
            v9 += 32;
            v13 += 4;
            if ( v11 == v9 )
              return v4;
          }
        }
      }
    }
  }
  else
  {
    v3 = *(_WORD *)(a1 + 2) & 0x3F;
    if ( v3 == (unsigned int)sub_B52F50(*(_WORD *)(a2 + 2) & 0x3F) && *(_QWORD *)(a1 - 64) == *(_QWORD *)(a2 - 32) )
    {
      LOBYTE(v4) = *(_QWORD *)(a1 - 32) == *(_QWORD *)(a2 - 64);
      return v4;
    }
  }
  return 0;
}
