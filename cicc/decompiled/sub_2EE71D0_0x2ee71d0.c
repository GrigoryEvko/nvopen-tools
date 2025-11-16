// Function: sub_2EE71D0
// Address: 0x2ee71d0
//
bool __fastcall sub_2EE71D0(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // edx
  __int64 v4; // rax
  int v5; // r13d
  __int64 v6; // rbx
  __int64 v7; // r12
  int v8; // r15d
  int v9; // eax
  int v10; // [rsp+4h] [rbp-3Ch]
  __int64 v11; // [rsp+8h] [rbp-38h]

  if ( !*(_WORD *)(a1 + 68) )
    return (unsigned int)sub_2E8B030(a1) != 0;
  if ( *(_WORD *)(a1 + 68) != 68 )
    return 0;
  v2 = sub_2E88D60(a1);
  v3 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  v11 = *(_QWORD *)(v2 + 32);
  v4 = *(_QWORD *)(a1 + 32);
  v5 = *(_DWORD *)(v4 + 8);
  if ( v3 > 1 )
  {
    v10 = 0;
    v6 = 40;
    v7 = 80LL * ((v3 - 2) >> 1) + 120;
    while ( 1 )
    {
      v8 = *(_DWORD *)(v4 + v6 + 8);
      if ( v5 != v8 )
      {
        v9 = *(unsigned __int16 *)(sub_2EBEE10(v11, v8) + 68);
        if ( v9 != 67 && v9 != 10 )
        {
          if ( v10 && v8 != v10 )
            return 0;
          v10 = v8;
        }
      }
      v6 += 80;
      if ( v7 == v6 )
        return 1;
      v4 = *(_QWORD *)(a1 + 32);
    }
  }
  return 1;
}
