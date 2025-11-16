// Function: sub_1A695F0
// Address: 0x1a695f0
//
__int64 __fastcall sub_1A695F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rax

  v11 = sub_15A9650(*(_QWORD *)(a1 + 160), *a6);
  v12 = *(_DWORD *)(a3 + 32);
  if ( v12 > 0x40 )
    v13 = **(_QWORD **)(a3 + 24);
  else
    v13 = (__int64)(*(_QWORD *)(a3 + 24) << (64 - (unsigned __int8)v12)) >> (64 - (unsigned __int8)v12);
  v14 = sub_159C470(v11, a5 * v13, 1u);
  return sub_1A69110(a1, 3, a2, v14, a4, (__int64)a6);
}
