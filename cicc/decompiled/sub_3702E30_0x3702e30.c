// Function: sub_3702E30
// Address: 0x3702e30
//
__int64 __fastcall sub_3702E30(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rax
  __int16 v9; // ax
  __int64 result; // rax
  _WORD v11[2]; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v12; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v13[6]; // [rsp+10h] [rbp-30h] BYREF

  *(_DWORD *)(a1 + 32) = a2;
  v6 = *(_QWORD *)(a1 + 48);
  *(_BYTE *)(a1 + 36) = 1;
  if ( v6 != *(_QWORD *)(a1 + 56) )
    *(_QWORD *)(a1 + 56) = v6;
  v7 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 8) = 0;
  v8 = 0;
  *(_QWORD *)(a1 + 136) = 0;
  if ( !v7 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), 1u, 4u, a5, a6);
    v8 = 4LL * *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + v8) = 0;
  ++*(_DWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 240) = 12;
  if ( a2 )
  {
    *(_QWORD *)(a1 + 232) = &qword_5041648;
    v9 = 4614;
  }
  else
  {
    *(_QWORD *)(a1 + 232) = &qword_5041658;
    v9 = 4611;
  }
  v11[1] = v9;
  v11[0] = 2;
  v13[0] = v11;
  v13[1] = 4;
  sub_370EAB0(&v12, a1 + 144, v13);
  if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  result = sub_3719260(&v12, a1 + 80, v11, 4);
  if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  return result;
}
