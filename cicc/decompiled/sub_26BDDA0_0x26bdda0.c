// Function: sub_26BDDA0
// Address: 0x26bdda0
//
char __fastcall sub_26BDDA0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r15
  unsigned __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r12
  unsigned int v8; // eax
  unsigned int v9; // eax
  bool v10; // cf
  unsigned int v11; // eax
  __int64 v13; // [rsp-40h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 32);
  if ( v2 == *(_DWORD *)(a2 + 32) )
  {
    if ( v2 )
    {
      v3 = *(_QWORD *)(a2 + 24);
      v4 = *(_QWORD *)(a1 + 24);
      v5 = v4;
      if ( v3 <= v4 )
        v5 = *(_QWORD *)(a2 + 24);
      if ( v5 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_QWORD *)(a2 + 16);
        v13 = 0;
        while ( 1 )
        {
          v8 = sub_C1F8C0(v6, v7);
          if ( v8 )
            return v8 >> 31;
          v9 = *(_DWORD *)(v7 + 16);
          v10 = *(_DWORD *)(v6 + 16) < v9;
          if ( *(_DWORD *)(v6 + 16) != v9
            || (v11 = *(_DWORD *)(v7 + 20), v10 = *(_DWORD *)(v6 + 20) < v11, *(_DWORD *)(v6 + 20) != v11) )
          {
            LOBYTE(v2) = v10;
            return v2;
          }
          ++v13;
          v6 += 24;
          v7 += 24;
          if ( v5 == v13 )
            goto LABEL_15;
        }
      }
      else
      {
LABEL_15:
        LOBYTE(v2) = v3 > v4;
      }
    }
    else
    {
      return (unsigned int)sub_C1F8C0(a1, a2) >> 31;
    }
  }
  else
  {
    LOBYTE(v2) = v2 < *(_DWORD *)(a2 + 32);
  }
  return v2;
}
