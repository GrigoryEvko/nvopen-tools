// Function: sub_C472A0
// Address: 0xc472a0
//
__int64 __fastcall sub_C472A0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v4; // r13d
  unsigned __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v9; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v5 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a2 + 8);
  if ( v4 > 0x40 )
  {
    v9 = sub_2207820(8 * (((unsigned __int64)v4 + 63) >> 6));
    sub_C47210(v9, *(_QWORD *)a2, *a3, ((unsigned __int64)*(unsigned int *)(a2 + 8) + 63) >> 6);
    *(_DWORD *)(a1 + 8) = v4;
    *(_QWORD *)a1 = v9;
    *(_QWORD *)(v9 + 8LL * ((unsigned int)(((unsigned __int64)v4 + 63) >> 6) - 1)) &= v5;
  }
  else
  {
    v6 = *a3 * *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = v4;
    v7 = v6 & v5;
    if ( !v4 )
      v7 = 0;
    *(_QWORD *)a1 = v7;
  }
  return a1;
}
