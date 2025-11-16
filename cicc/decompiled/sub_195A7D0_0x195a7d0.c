// Function: sub_195A7D0
// Address: 0x195a7d0
//
__int64 __fastcall sub_195A7D0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // r12
  int v11; // ebx
  unsigned int v12; // ebx
  __int64 v13; // r15
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // r14
  unsigned __int64 v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 result; // rax
  __int64 v21; // [rsp+18h] [rbp-58h]
  unsigned __int8 v22; // [rsp+18h] [rbp-58h]
  _QWORD v23[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v24[8]; // [rsp+30h] [rbp-40h] BYREF

  v10 = v24;
  v11 = *(_DWORD *)(a2 + 20);
  v21 = *(_QWORD *)(a2 + 40);
  v23[0] = v24;
  v23[1] = 0x100000001LL;
  v24[0] = 0;
  v12 = v11 & 0xFFFFFFF;
  if ( !v12 )
    return 0;
  v13 = 0;
  v14 = 8LL * v12;
  while ( 1 )
  {
    v15 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v16 = *(_QWORD *)(v13 + v15 + 24LL * *(unsigned int *)(a2 + 56) + 8);
    v17 = sub_157EBA0(v16);
    if ( *(_BYTE *)(v17 + 16) == 26 && (*(_DWORD *)(v17 + 20) & 0xFFFFFFF) == 1 )
    {
      *v10 = v16;
      result = sub_195A750(a1, v21, (__int64)v23, a3, a4, a5, a6, v18, v19, a9, a10);
      v10 = (__int64 *)v23[0];
      if ( (_BYTE)result )
        break;
    }
    v13 += 8;
    if ( v14 == v13 )
    {
      result = 0;
      break;
    }
  }
  if ( v10 != v24 )
  {
    v22 = result;
    _libc_free((unsigned __int64)v10);
    return v22;
  }
  return result;
}
