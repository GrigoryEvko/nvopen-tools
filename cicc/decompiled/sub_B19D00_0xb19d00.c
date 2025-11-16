// Function: sub_B19D00
// Address: 0xb19d00
//
__int64 __fastcall sub_B19D00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 v4; // r8
  unsigned int v5; // eax
  unsigned int v6; // ecx
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 v9; // r10
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v13[3]; // [rsp-18h] [rbp-18h] BYREF

  if ( a3 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    v5 = *(_DWORD *)(a3 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = *(_DWORD *)(a1 + 32);
  if ( v5 >= v6 )
    return 1;
  v7 = *(_QWORD *)(a1 + 24);
  if ( !*(_QWORD *)(v7 + 8 * v4) )
    return 1;
  v8 = *(_QWORD *)(a2 + 40);
  if ( v8 )
  {
    v9 = (unsigned int)(*(_DWORD *)(v8 + 44) + 1);
    v10 = *(_DWORD *)(v8 + 44) + 1;
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  if ( v6 <= v10 || !*(_QWORD *)(v7 + 8 * v9) || a3 == v8 )
    return 0;
  if ( *(_BYTE *)a2 != 34 )
    return sub_B19720(a1, *(_QWORD *)(a2 + 40), a3);
  v13[2] = v3;
  v11 = *(_QWORD *)(a2 - 96);
  v13[0] = v8;
  v13[1] = v11;
  return sub_B19C20(a1, v13, a3);
}
