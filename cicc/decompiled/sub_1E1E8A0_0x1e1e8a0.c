// Function: sub_1E1E8A0
// Address: 0x1e1e8a0
//
__int64 __fastcall sub_1E1E8A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r13
  __int64 v4; // r15
  __int64 v5; // rax
  int v6; // r12d
  unsigned __int8 v7; // al
  __int64 (*v8)(); // r15
  __int64 v9; // rax
  __int64 v11; // [rsp+0h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 32);
  for ( i = v2 + 40LL * *(unsigned int *)(a2 + 40); i != v2; v2 += 40 )
  {
    if ( *(_BYTE *)v2 )
      continue;
    v6 = *(_DWORD *)(v2 + 8);
    if ( !v6 )
      continue;
    if ( v6 > 0 )
    {
      v7 = *(_BYTE *)(v2 + 3);
      if ( (v7 & 0x10) == 0 )
      {
        if ( !(unsigned __int8)sub_1E69FD0(a1[33]) )
        {
          v11 = a1[31];
          v8 = *(__int64 (**)())(*(_QWORD *)v11 + 80LL);
          v9 = sub_1E15F70(a2);
          if ( v8 == sub_1E1C7F0
            || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))v8)(v11, (unsigned int)v6, v9) )
          {
            return 0;
          }
        }
        continue;
      }
      if ( (((v7 & 0x10) != 0) & (v7 >> 6)) == 0 || sub_1DD6670(**(_QWORD **)(a1[76] + 32LL), v6, -1) )
        return 0;
    }
    if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
    {
      v4 = a1[76];
      v5 = sub_1E69D00(a1[33], (unsigned int)v6);
      if ( sub_1DA1810(v4 + 56, *(_QWORD *)(v5 + 24)) )
        return 0;
    }
  }
  return 1;
}
