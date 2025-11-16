// Function: sub_155C2B0
// Address: 0x155c2b0
//
__int64 __fastcall sub_155C2B0(__int64 a1, __int64 a2, char a3)
{
  unsigned __int8 v4; // al
  char v5; // r15
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // rax
  __int64 *v9; // rcx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v13[12]; // [rsp+0h] [rbp-60h] BYREF

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 <= 0x17u )
  {
    v5 = v4 == 0 || v4 == 19;
  }
  else
  {
    v5 = 0;
    if ( v4 == 78 )
    {
      v6 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v6 + 16) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
      {
        v7 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        {
          v8 = *(__int64 **)(a1 - 8);
          v9 = &v8[v7];
        }
        else
        {
          v9 = (__int64 *)a1;
          v8 = (__int64 *)(a1 - v7 * 8);
        }
        if ( v8 == v9 )
        {
LABEL_15:
          v5 = 0;
        }
        else
        {
          while ( 1 )
          {
            v10 = *v8;
            if ( *v8 )
            {
              if ( *(_BYTE *)(v10 + 16) == 19 && (unsigned __int8)(**(_BYTE **)(v10 + 24) - 4) <= 0x1Eu )
                break;
            }
            v8 += 3;
            if ( v9 == v8 )
              goto LABEL_15;
          }
          v5 = 1;
        }
      }
    }
  }
  v11 = sub_1548BC0(a1);
  sub_154BA10((__int64)v13, (__int64)v11, v5);
  sub_155BD40(a1, a2, (__int64)v13, a3);
  return sub_154BA40(v13);
}
