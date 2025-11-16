// Function: sub_B30000
// Address: 0xb30000
//
__int64 __fastcall sub_B30000(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        char a4,
        char a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int16 a9,
        __int64 a10,
        char a11)
{
  unsigned int v11; // eax
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // rdx
  __int64 v17; // rax

  if ( BYTE4(a10) )
    v11 = a10;
  else
    v11 = *(_DWORD *)(a2 + 324);
  v12 = a1 + 56;
  sub_B2FEA0(a1, a3, a4, a5, a6, a7, a9, v11, a11);
  if ( a8 )
  {
    sub_BA85C0(*(_QWORD *)(a8 + 40) + 8LL, a1);
    v13 = *(_QWORD *)(a8 + 56);
    v14 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 64) = a8 + 56;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 56) = v13 | v14 & 7;
    *(_QWORD *)(v13 + 8) = v12;
    result = v12 | *(_QWORD *)(a8 + 56) & 7LL;
    *(_QWORD *)(a8 + 56) = result;
  }
  else
  {
    sub_BA85C0(a2 + 8, a1);
    v16 = *(_QWORD *)(a2 + 8);
    v17 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 64) = a2 + 8;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 56) = v16 | v17 & 7;
    *(_QWORD *)(v16 + 8) = v12;
    result = v12 | *(_QWORD *)(a2 + 8) & 7LL;
    *(_QWORD *)(a2 + 8) = result;
  }
  return result;
}
