// Function: sub_77EFA0
// Address: 0x77efa0
//
__int64 __fastcall sub_77EFA0(__int64 a1, __int64 a2, __int64 a3, __m128i **a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r15
  _QWORD *v10; // rax
  __m128i *v11; // rsi
  unsigned int v12; // r12d
  __int64 v14; // rax
  int v15; // edi
  int v16; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v9 = *(_QWORD *)(*(_QWORD *)(a2 + 240) + 32LL);
  v10 = sub_724DC0();
  v11 = *a4;
  v17[0] = (__int64)v10;
  if ( sub_77D750(a1, v11, (__int64)v11, v9, (__int64)v10) )
  {
    v12 = 1;
    sub_7296C0(&v16);
    v14 = sub_724E50(v17, v11);
    *(_BYTE *)a5 = 2;
    v15 = v16;
    *(_QWORD *)(a5 + 8) = v14;
    *(_DWORD *)(a5 + 16) = 0;
    v17[0] = v14;
    sub_729730(v15);
    *(_BYTE *)(a6 + -(((unsigned int)(a5 - a6) >> 3) + 10)) |= 1 << ((a5 - a6) & 7);
  }
  else
  {
    v12 = 0;
    sub_724E30((__int64)v17);
  }
  return v12;
}
