// Function: sub_FD7FB0
// Address: 0xfd7fb0
//
char __fastcall sub_FD7FB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx

  if ( *(_BYTE *)a2 == 85 )
  {
    v9 = *(_QWORD *)(a2 - 32);
    if ( v9 )
    {
      if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
      {
        v4 = *(_DWORD *)(v9 + 36) - 68;
        if ( v4 <= 3 )
          return v4;
      }
      if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
      {
        v4 = *(_DWORD *)(v9 + 36);
        if ( v4 == 155 )
          return v4;
        if ( v4 > 0x9B )
        {
          if ( v4 == 291 || v4 == 324 )
            return v4;
        }
        else if ( v4 > 6 )
        {
          if ( v4 == 11 )
            return v4;
        }
        else if ( v4 > 4 )
        {
          return v4;
        }
      }
    }
  }
  if ( (unsigned __int8)sub_B46420(a2) || (LOBYTE(v4) = sub_B46490(a2), (_BYTE)v4) )
  {
    v3 = sub_FD7B50((__int64 **)a1, (unsigned __int8 *)a2);
    if ( v3 )
    {
      LOBYTE(v4) = sub_FD7E80(v3, a2);
    }
    else
    {
      v5 = sub_22077B0(72);
      if ( v5 )
      {
        *(_DWORD *)(v5 + 64) &= 0x80000000;
        v6 = 0;
        *(_QWORD *)(v5 + 16) = 0;
        *(_QWORD *)(v5 + 24) = v5 + 40;
        *(_QWORD *)(v5 + 32) = 0;
        *(_QWORD *)(v5 + 40) = 0;
        *(_QWORD *)(v5 + 48) = 0;
        *(_QWORD *)(v5 + 56) = 0;
      }
      else
      {
        v6 = MEMORY[0] & 7;
      }
      v7 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(v5 + 8) = a1 + 8;
      v7 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v5 = v7 | v6;
      *(_QWORD *)(v7 + 8) = v5;
      v8 = *(_QWORD *)(a1 + 8) & 7LL | v5;
      *(_QWORD *)(a1 + 8) = v8;
      LOBYTE(v4) = sub_FD7E80(v8 & 0xFFFFFFFFFFFFFFF8LL, a2);
    }
  }
  return v4;
}
