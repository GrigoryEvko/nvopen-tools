// Function: sub_21C2AA0
// Address: 0x21c2aa0
//
__int64 __fastcall sub_21C2AA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        unsigned __int8 a10)
{
  unsigned int v10; // r12d
  __int64 *v12; // rdx
  __int64 v13; // r15
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // r10
  __int64 v19; // rax
  _QWORD *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  int v23; // edx
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  int v26; // [rsp+28h] [rbp-38h]

  v10 = 0;
  if ( *(_WORD *)(a3 + 24) == 52 )
  {
    v12 = *(__int64 **)(a3 + 32);
    v13 = v12[5];
    LOBYTE(v10) = *(_WORD *)(v13 + 24) == 10 || *(_WORD *)(v13 + 24) == 32;
    if ( (_BYTE)v10 )
    {
      LOBYTE(v16) = sub_21C2A00(a1, *v12, v12[1], a8);
      v10 = v16;
      if ( (_BYTE)v16 )
      {
        v17 = *(_QWORD *)(a2 + 72);
        v18 = *(_QWORD *)(a1 + 272);
        v25 = v17;
        if ( v17 )
        {
          v24 = v18;
          sub_1623A60((__int64)&v25, v17, 2);
          v18 = v24;
        }
        v26 = *(_DWORD *)(a2 + 64);
        v19 = *(_QWORD *)(v13 + 88);
        v20 = *(_QWORD **)(v19 + 24);
        if ( *(_DWORD *)(v19 + 32) > 0x40u )
          v20 = (_QWORD *)*v20;
        v21 = sub_1D38BB0(v18, (__int64)v20, (__int64)&v25, a10, 0, 1, a4, a5, a6, 0);
        v22 = v25;
        *(_QWORD *)a9 = v21;
        *(_DWORD *)(a9 + 8) = v23;
        if ( v22 )
          sub_161E7C0((__int64)&v25, v22);
      }
    }
  }
  return v10;
}
