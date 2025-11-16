// Function: sub_15E5440
// Address: 0x15e5440
//
char __fastcall sub_15E5440(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // rdx

  if ( a2 )
  {
    if ( sub_15E4F60(a1) )
      *(_DWORD *)(a1 + 20) = *(_DWORD *)(a1 + 20) & 0xF0000000 | 1;
    if ( *(_QWORD *)(a1 - 24) )
    {
      v2 = *(_QWORD *)(a1 - 16);
      v3 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v3 = v2;
      if ( v2 )
        *(_QWORD *)(v2 + 16) = *(_QWORD *)(v2 + 16) & 3LL | v3;
    }
    *(_QWORD *)(a1 - 24) = a2;
    v4 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 16) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = (a1 - 16) | *(_QWORD *)(v4 + 16) & 3LL;
    v5 = *(_QWORD *)(a1 - 8) & 3LL | (a2 + 8);
    *(_QWORD *)(a1 - 8) = v5;
    *(_QWORD *)(a2 + 8) = a1 - 24;
  }
  else
  {
    LOBYTE(v5) = sub_15E4F60(a1);
    if ( !(_BYTE)v5 )
    {
      if ( *(_QWORD *)(a1 - 24) )
      {
        v6 = *(_QWORD *)(a1 - 16);
        v5 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v5 = v6;
        if ( v6 )
        {
          v5 |= *(_QWORD *)(v6 + 16) & 3LL;
          *(_QWORD *)(v6 + 16) = v5;
        }
      }
      *(_DWORD *)(a1 + 20) &= 0xF0000000;
      *(_QWORD *)(a1 - 24) = 0;
    }
  }
  return v5;
}
