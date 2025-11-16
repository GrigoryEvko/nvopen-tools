// Function: sub_15E46C0
// Address: 0x15e46c0
//
__int64 __fastcall sub_15E46C0(__int64 a1)
{
  __int64 v1; // rbp
  __int64 result; // rax
  int v3; // esi
  __int64 v4; // rax
  int v5; // eax
  _QWORD v6[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return 0;
  v6[3] = v1;
  v3 = *(_DWORD *)(a1 + 32);
  v6[0] = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 112LL);
  result = sub_1560290(v6, v3, 32);
  if ( !(_BYTE)result )
  {
    if ( sub_15E0380(a1) )
    {
      v4 = *(_QWORD *)a1;
      if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
        v4 = **(_QWORD **)(v4 + 16);
      LOBYTE(v5) = sub_15E4690(*(_QWORD *)(a1 + 24), *(_DWORD *)(v4 + 8) >> 8);
      return v5 ^ 1u;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
