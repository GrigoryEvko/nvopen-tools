// Function: sub_2C1B5C0
// Address: 0x2c1b5c0
//
__int64 __fastcall sub_2C1B5C0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v9; // [rsp+8h] [rbp-58h]
  __int64 v10[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v11; // [rsp+30h] [rbp-30h]

  v10[0] = *(_QWORD *)(a1 + 88);
  if ( v10[0] )
    sub_2AAAFA0(v10);
  sub_2BF1A90(a2, (__int64)v10);
  sub_9C6650(v10);
  if ( (unsigned int)(*(_DWORD *)(a1 + 152) - 38) > 2 )
    BUG();
  v3 = *(__int64 **)(a1 + 48);
  BYTE4(v10[0]) = 0;
  LODWORD(v10[0]) = 0;
  v4 = sub_2BFB120(a2, *v3, (unsigned int *)v10);
  v5 = *(_QWORD *)(a1 + 160);
  v6 = *(_QWORD *)(a2 + 904);
  v7 = *(_DWORD *)(a1 + 152);
  v11 = 257;
  return sub_2C13D90(v6, v7, v4, v5, (__int64)v10, 0, v9);
}
