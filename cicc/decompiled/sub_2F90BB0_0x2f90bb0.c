// Function: sub_2F90BB0
// Address: 0x2f90bb0
//
__int64 __fastcall sub_2F90BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rbx
  unsigned int v15; // r14d
  __int64 v16; // r15

  sub_2F90AB0(a1, a2, a3, a4, a5, a6);
  v10 = sub_2F90B20(a1, a3, a2, v7, v8, v9);
  if ( (_BYTE)v10 )
    return 1;
  v14 = *(_QWORD *)(a2 + 40);
  v15 = v10;
  v16 = v14 + 16LL * *(unsigned int *)(a2 + 48);
  if ( v16 != v14 )
  {
    while ( (*(_QWORD *)v14 & 6) != 0
         || !*(_DWORD *)(v14 + 8)
         || !(unsigned __int8)sub_2F90B20(a1, a3, *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL, v11, v12, v13) )
    {
      v14 += 16;
      if ( v16 == v14 )
        return v15;
    }
    return 1;
  }
  return v15;
}
