// Function: sub_2024CF0
// Address: 0x2024cf0
//
unsigned __int64 __fastcall sub_2024CF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rsi
  int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // edx
  unsigned __int64 result; // rax
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  int v17; // [rsp+28h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 72);
  v16 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v16, v10, 2);
  v17 = *(_DWORD *)(a2 + 64);
  sub_2017DE0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), (_DWORD *)a3, (_DWORD *)a4);
  *(_QWORD *)a3 = sub_1D332F0(
                    *(__int64 **)(a1 + 8),
                    167,
                    (__int64)&v16,
                    *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a3 + 40LL) + 16LL * *(unsigned int *)(a3 + 8)),
                    *(const void ***)(*(_QWORD *)(*(_QWORD *)a3 + 40LL) + 16LL * *(unsigned int *)(a3 + 8) + 8),
                    0,
                    a5,
                    a6,
                    a7,
                    *(_QWORD *)a3,
                    *(_QWORD *)(a3 + 8),
                    *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  *(_DWORD *)(a3 + 8) = v11;
  v12 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          167,
          (__int64)&v16,
          *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * *(unsigned int *)(a4 + 8)),
          *(const void ***)(*(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * *(unsigned int *)(a4 + 8) + 8),
          0,
          a5,
          a6,
          a7,
          *(_QWORD *)a4,
          *(_QWORD *)(a4 + 8),
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  v13 = v16;
  *(_QWORD *)a4 = v12;
  result = v14;
  *(_DWORD *)(a4 + 8) = v14;
  if ( v13 )
    return sub_161E7C0((__int64)&v16, v13);
  return result;
}
