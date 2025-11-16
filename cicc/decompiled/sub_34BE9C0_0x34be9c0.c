// Function: sub_34BE9C0
// Address: 0x34be9c0
//
__int64 __fastcall sub_34BE9C0(__int64 a1)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  int v5; // ecx
  __int64 result; // rax
  bool v7; // r8
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax

  v2 = *(_QWORD *)(a1 + 48);
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    BUG();
  v4 = *(_QWORD *)v3;
  v5 = *(_DWORD *)(v3 + 44);
  if ( (*(_QWORD *)v3 & 4) != 0 )
  {
    if ( (v5 & 4) != 0 )
    {
LABEL_4:
      result = 0;
      if ( (*(_BYTE *)(*(_QWORD *)(v3 + 16) + 24LL) & 0x20) != 0 )
        return result;
      v8 = v2 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_15:
      v9 = *(_QWORD *)v8;
      v10 = *(_DWORD *)(v8 + 44);
      if ( (*(_QWORD *)v8 & 4) != 0 )
      {
        if ( (v10 & 4) != 0 )
          goto LABEL_17;
      }
      else if ( (v10 & 4) != 0 )
      {
        while ( 1 )
        {
          v8 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          LOBYTE(v10) = *(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 44);
          if ( (v10 & 4) == 0 )
            break;
          v9 = *(_QWORD *)v8;
        }
      }
      if ( (v10 & 8) != 0 )
      {
        LOBYTE(v11) = sub_2E88A90(v8, 2048, 1);
        return (unsigned int)v11 ^ 1;
      }
LABEL_17:
      v11 = (*(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL) >> 11) & 1LL;
      return (unsigned int)v11 ^ 1;
    }
  }
  else if ( (v5 & 4) != 0 )
  {
    while ( 1 )
    {
      v3 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      LOBYTE(v5) = *(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 44);
      if ( (v5 & 4) == 0 )
        break;
      v4 = *(_QWORD *)v3;
    }
  }
  if ( (v5 & 8) == 0 )
    goto LABEL_4;
  v7 = sub_2E88A90(v3, 32, 1);
  result = 0;
  if ( !v7 )
  {
    v8 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v8 )
      BUG();
    goto LABEL_15;
  }
  return result;
}
