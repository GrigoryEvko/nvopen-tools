// Function: sub_F0C890
// Address: 0xf0c890
//
__int64 __fastcall sub_F0C890(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // [rsp-38h] [rbp-38h] BYREF
  __int64 v12; // [rsp-30h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 8) != 12 )
    return 0;
  if ( *(_BYTE *)(a3 + 8) != 12 )
    return 0;
  v5 = sub_BCAE30(a2);
  v12 = v6;
  v11 = v5;
  v7 = sub_CA1930(&v11);
  v8 = sub_BCAE30(a3);
  v12 = v9;
  v11 = v8;
  v10 = sub_CA1930(&v11);
  return sub_F0C790(a1, v7, v10);
}
