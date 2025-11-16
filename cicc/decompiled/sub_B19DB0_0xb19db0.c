// Function: sub_B19DB0
// Address: 0xb19db0
//
__int64 __fastcall sub_B19DB0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // bl
  __int64 v4; // r14
  __int64 v6; // rdx
  unsigned int v7; // eax
  __int64 v9; // [rsp+8h] [rbp-38h]

  v3 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 1;
  v4 = *(_QWORD *)(a3 + 40);
  if ( v4 )
  {
    v6 = (unsigned int)(*(_DWORD *)(v4 + 44) + 1);
    v7 = *(_DWORD *)(v4 + 44) + 1;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  if ( v7 >= *(_DWORD *)(a1 + 32) || !*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v6) )
    return 1;
  v9 = *(_QWORD *)(a2 + 40);
  if ( (unsigned __int8)sub_B192B0(a1, v9) != 1 || a2 == a3 )
    return 0;
  if ( v3 == 34 || v3 == 40 || *(_BYTE *)a3 == 84 )
    return sub_B19D00(a1, a2, v4);
  if ( v4 == v9 )
    return sub_B445A0(a2, a3);
  return sub_B19720(a1, v9, v4);
}
