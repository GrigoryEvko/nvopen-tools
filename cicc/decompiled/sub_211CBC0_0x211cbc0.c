// Function: sub_211CBC0
// Address: 0x211cbc0
//
unsigned __int64 __fastcall sub_211CBC0(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        _DWORD *a4,
        double a5,
        double a6,
        double a7)
{
  __int64 v10; // rsi
  const void ***v11; // rax
  int v12; // edx
  const void ***v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // edx
  unsigned __int64 result; // rax
  __int64 v18; // [rsp+20h] [rbp-40h] BYREF
  int v19; // [rsp+28h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 72);
  v18 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v18, v10, 2);
  v19 = *(_DWORD *)(a2 + 64);
  sub_2016B80(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4);
  v11 = (const void ***)(*(_QWORD *)(*(_QWORD *)a3 + 40LL) + 16LL * (unsigned int)a3[2]);
  *(_QWORD *)a3 = sub_1D309E0(
                    *(__int64 **)(a1 + 8),
                    162,
                    (__int64)&v18,
                    *(unsigned __int8 *)v11,
                    v11[1],
                    0,
                    a5,
                    a6,
                    a7,
                    *(_OWORD *)a3);
  a3[2] = v12;
  v13 = (const void ***)(*(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * (unsigned int)a4[2]);
  v14 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          162,
          (__int64)&v18,
          *(unsigned __int8 *)v13,
          v13[1],
          0,
          a5,
          a6,
          a7,
          *(_OWORD *)a4);
  v15 = v18;
  *(_QWORD *)a4 = v14;
  result = v16;
  a4[2] = v16;
  if ( v15 )
    return sub_161E7C0((__int64)&v18, v15);
  return result;
}
