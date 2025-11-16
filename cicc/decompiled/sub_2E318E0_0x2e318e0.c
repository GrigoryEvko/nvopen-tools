// Function: sub_2E318E0
// Address: 0x2e318e0
//
unsigned __int64 __fastcall sub_2E318E0(__int64 a1)
{
  __int64 v1; // r13
  unsigned __int64 v2; // r12
  __int64 v3; // r14
  __int64 v4; // rax
  unsigned __int64 v5; // rbx
  int v6; // eax
  int v7; // eax

  v1 = a1 + 48;
  v2 = a1 + 48;
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 != a1 + 48 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = *(_DWORD *)(v5 + 44);
      v2 = v5;
      if ( (v6 & 4) != 0 || (v6 & 8) == 0 )
        v4 = (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 24LL) >> 9) & 1LL;
      else
        LOBYTE(v4) = sub_2E88A90(v5, 512, 1);
      if ( !(_BYTE)v4 && (unsigned __int16)(*(_WORD *)(v5 + 68) - 14) > 4u )
        break;
      if ( v3 == v5 )
      {
        while ( 1 )
        {
LABEL_13:
          v7 = *(_DWORD *)(v2 + 44);
          if ( (v7 & 4) != 0 || (v7 & 8) == 0 )
          {
            if ( (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) & 0x200LL) != 0 )
              return v2;
          }
          else if ( (unsigned __int8)sub_2E88A90(v2, 512, 1) )
          {
            return v2;
          }
          v2 = *(_QWORD *)(v2 + 8);
          if ( v1 == v2 )
            return v2;
        }
      }
    }
    if ( v1 != v5 )
      goto LABEL_13;
  }
  return v2;
}
