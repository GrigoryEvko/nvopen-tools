// Function: sub_33E2390
// Address: 0x33e2390
//
__int64 __fastcall sub_33E2390(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  __int64 v5; // rbx
  int v6; // eax
  unsigned int v7; // r12d
  int v8; // eax
  int v9; // eax

  v5 = sub_33CF5B0(a2, a3);
  v6 = *(_DWORD *)(v5 + 24);
  if ( v6 != 11 && v6 != 35 )
  {
    v7 = sub_33CA6D0(v5);
    if ( !(_BYTE)v7 )
    {
      v8 = *(_DWORD *)(v5 + 24);
      if ( (unsigned int)(v8 - 13) > 1 && (unsigned int)(v8 - 37) > 1 || v8 != 13 )
        goto LABEL_7;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 1952LL))(
              *(_QWORD *)(a1 + 16),
              v5) )
      {
        v8 = *(_DWORD *)(v5 + 24);
LABEL_7:
        if ( v8 == 168 )
        {
          v9 = *(_DWORD *)(**(_QWORD **)(v5 + 40) + 24LL);
          LOBYTE(v7) = v9 == 11;
          LOBYTE(v9) = v9 == 35;
          v7 |= v9;
        }
        return v7;
      }
    }
    return 1;
  }
  if ( a4 )
    return 1;
  return ((*(_BYTE *)(v5 + 32) >> 3) ^ 1) & 1;
}
