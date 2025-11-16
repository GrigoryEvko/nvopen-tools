// Function: sub_1D016F0
// Address: 0x1d016f0
//
__int64 __fastcall sub_1D016F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _DWORD *a5, _DWORD *a6, __int64 a7)
{
  __int64 v9; // r13
  __int64 (__fastcall *v10)(__int64, unsigned __int8); // rax
  __int64 v11; // rax
  __int64 (__fastcall *v12)(__int64, unsigned __int8); // rax
  __int64 result; // rax
  __int64 v14; // rsi
  __int16 v15; // ax
  int v17; // edx
  __int64 v18; // rax
  const __m128i *v19; // rax
  int v20; // edx
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _OWORD v23[6]; // [rsp+0h] [rbp-60h] BYREF

  v9 = *(unsigned __int8 *)(a1 + 24);
  if ( (_BYTE)v9 == 113 )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v15 = *(_WORD *)(v14 + 24);
    if ( v15 == 47 )
    {
      result = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a7 + 40) + 24LL)
                                                           + 16LL
                                                           * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v14 + 32) + 40LL)
                                                                        + 84LL)
                                                            & 0x7FFFFFFF))
                                               & 0xFFFFFFFFFFFFFFF8LL)
                                   + 24LL);
      *a5 = result;
      *a6 = 1;
    }
    else
    {
      v17 = v15;
      v18 = (unsigned int)~v15;
      if ( v17 == -15 )
      {
        v21 = *(_QWORD *)(**(_QWORD **)(v14 + 32) + 88LL);
        v22 = *(_QWORD **)(v21 + 24);
        if ( *(_DWORD *)(v21 + 32) > 0x40u )
          v22 = (_QWORD *)*v22;
        result = *(unsigned __int16 *)(**(_QWORD **)(*(_QWORD *)(a4 + 256) + 8LL * (unsigned int)v22) + 24LL);
        *a5 = result;
        *a6 = 1;
      }
      else
      {
        v19 = (const __m128i *)(*(_QWORD *)(a3 + 8) + (v18 << 6));
        v20 = *(_DWORD *)(a1 + 16);
        v23[0] = _mm_loadu_si128(v19);
        v23[1] = _mm_loadu_si128(v19 + 1);
        v23[2] = _mm_loadu_si128(v19 + 2);
        v23[3] = _mm_loadu_si128(v19 + 3);
        result = *(unsigned __int16 *)(*(_QWORD *)((__int64 (__fastcall *)(__int64, _OWORD *, _QWORD, __int64, __int64))sub_1F3AD60)(
                                                    a3,
                                                    v23,
                                                    (unsigned int)(v20 - 1),
                                                    a4,
                                                    a7)
                                     + 24LL);
        *a5 = result;
        *a6 = 1;
      }
    }
  }
  else
  {
    v10 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)a2 + 296LL);
    if ( v10 == sub_1D00B40 )
      v11 = *(_QWORD *)(a2 + 8LL * (unsigned __int8)v9 + 1272);
    else
      v11 = v10(a2, *(_BYTE *)(a1 + 24));
    *a5 = *(unsigned __int16 *)(*(_QWORD *)v11 + 24LL);
    v12 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)a2 + 304LL);
    if ( v12 == sub_1D00B50 )
      LOBYTE(result) = *(_BYTE *)(a2 + v9 + 2192);
    else
      LOBYTE(result) = v12(a2, v9);
    result = (unsigned __int8)result;
    *a6 = (unsigned __int8)result;
  }
  return result;
}
