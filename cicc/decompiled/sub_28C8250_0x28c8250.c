// Function: sub_28C8250
// Address: 0x28c8250
//
__int64 __fastcall sub_28C8250(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // edx
  __int64 v4; // r8
  int v5; // edi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  int v9; // edx
  int v10; // r10d

  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    if ( *(_BYTE *)a2 != 28 )
      BUG();
    return *(_QWORD *)(a2 + 64);
  }
  else
  {
    result = *(_QWORD *)(a2 + 40);
    if ( !result )
    {
      v3 = *(_DWORD *)(a1 + 1656);
      v4 = *(_QWORD *)(a1 + 1640);
      if ( v3 )
      {
        v5 = v3 - 1;
        v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v7 = (__int64 *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
        {
          return v7[1];
        }
        else
        {
          v9 = 1;
          while ( v8 != -4096 )
          {
            v10 = v9 + 1;
            v6 = v5 & (v9 + v6);
            v7 = (__int64 *)(v4 + 16LL * v6);
            v8 = *v7;
            if ( a2 == *v7 )
              return v7[1];
            v9 = v10;
          }
        }
      }
    }
  }
  return result;
}
