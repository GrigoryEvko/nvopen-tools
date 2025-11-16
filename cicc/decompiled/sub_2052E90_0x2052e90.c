// Function: sub_2052E90
// Address: 0x2052e90
//
__int64 __fastcall sub_2052E90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned int v7; // r12d
  unsigned __int64 v8; // rdi
  unsigned int v9; // eax
  unsigned int v10; // [rsp-1Ch] [rbp-1Ch] BYREF
  __int64 v11; // [rsp-8h] [rbp-8h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 712) + 32LL);
  if ( v5 )
    return sub_13774B0(v5, v4, *(_QWORD *)(a3 + 40));
  v11 = v3;
  v7 = 1;
  v8 = sub_157EBA0(v4);
  if ( v8 )
  {
    v9 = sub_15F4D60(v8);
    if ( v9 )
      v7 = v9;
  }
  sub_16AF710(&v10, 1u, v7);
  return v10;
}
