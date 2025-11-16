// Function: sub_2FD62C0
// Address: 0x2fd62c0
//
__int64 __fastcall sub_2FD62C0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r12
  int v3; // eax
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rax

  if ( *(_DWORD *)(a1 + 120) != 1 )
    return 0;
  if ( !*(_DWORD *)(a1 + 72) )
    return 0;
  v2 = sub_2E319B0(a1, 1);
  result = 1;
  if ( v2 == a1 + 48 )
    return result;
  v3 = *(_DWORD *)(v2 + 44);
  if ( (v3 & 4) == 0 && (v3 & 8) != 0 )
    LOBYTE(v4) = sub_2E88A90(v2, 1024, 1);
  else
    v4 = (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) >> 10) & 1LL;
  if ( !(_BYTE)v4 )
    return 0;
  v5 = *(_DWORD *)(v2 + 44);
  if ( (v5 & 4) != 0 || (v5 & 8) == 0 )
    v6 = (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) >> 8) & 1LL;
  else
    LOBYTE(v6) = sub_2E88A90(v2, 256, 1);
  if ( !(_BYTE)v6 )
    return 0;
  v7 = *(_DWORD *)(v2 + 44);
  if ( (v7 & 4) != 0 || (v7 & 8) == 0 )
    v8 = (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) >> 11) & 1LL;
  else
    LOBYTE(v8) = sub_2E88A90(v2, 2048, 1);
  return (unsigned int)v8 ^ 1;
}
