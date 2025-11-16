// Function: sub_ADD600
// Address: 0xadd600
//
__int64 __fastcall sub_ADD600(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9,
        unsigned __int8 a10,
        unsigned __int8 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        int a15,
        __int64 a16)
{
  __int64 v16; // r10
  int v18; // ecx
  __int64 v19; // r14
  int v20; // eax
  int v21; // edx
  int v22; // eax
  __int64 v23; // r12
  __int64 result; // rax
  __int64 v25; // rdx
  int v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v16 = a3;
  v18 = 0;
  v19 = *(_QWORD *)(a1 + 8);
  if ( a6 )
  {
    v20 = sub_B9B140(v19, a5, a6);
    v16 = a3;
    v18 = v20;
  }
  v21 = 0;
  if ( a4 )
  {
    v27 = v18;
    v22 = sub_B9B140(v19, v16, a4);
    v18 = v27;
    v21 = v22;
  }
  v23 = sub_B0B820(v19, a2, v21, v18, a7, a8, a9, a10, a11, a13, a14, a15, a16, 1, 1);
  if ( !a12 )
    a12 = sub_ADD5E0(a1, 0, 0);
  result = sub_B0EF30(*(_QWORD *)(a1 + 8), v23, a12, 0, 1);
  v25 = *(unsigned int *)(a1 + 208);
  if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
  {
    v28 = result;
    sub_C8D5F0(a1 + 200, a1 + 216, v25 + 1, 8);
    v25 = *(unsigned int *)(a1 + 208);
    result = v28;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v25) = result;
  ++*(_DWORD *)(a1 + 208);
  return result;
}
