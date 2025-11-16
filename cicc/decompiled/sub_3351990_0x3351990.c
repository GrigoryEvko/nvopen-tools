// Function: sub_3351990
// Address: 0x3351990
//
__int64 __fastcall sub_3351990(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, _DWORD *a5, _DWORD *a6, __int64 a7)
{
  __int64 v8; // r13
  __int64 (__fastcall *v10)(__int64, unsigned __int16); // rax
  __int64 v11; // rax
  __int64 (__fastcall *v12)(__int64, unsigned __int16); // rax
  __int64 result; // rax
  __int64 v14; // rsi
  int v15; // eax
  __int64 v16; // rdx
  _QWORD *v17; // rax

  v8 = *(unsigned __int16 *)(a1 + 24);
  if ( (_WORD)v8 == 264 )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v15 = *(_DWORD *)(v14 + 24);
    if ( v15 == 50 )
    {
      result = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a7 + 32) + 56LL)
                                                           + 16LL
                                                           * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v14 + 40) + 40LL)
                                                                        + 96LL)
                                                            & 0x7FFFFFFF))
                                               & 0xFFFFFFFFFFFFFFF8LL)
                                   + 24LL);
      *a5 = result;
      *a6 = 1;
    }
    else if ( v15 == -20 )
    {
      v16 = *(_QWORD *)(**(_QWORD **)(v14 + 40) + 96LL);
      v17 = *(_QWORD **)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) > 0x40u )
        v17 = (_QWORD *)*v17;
      result = *(unsigned __int16 *)(**(_QWORD **)(*(_QWORD *)(a4 + 280) + 8LL * (unsigned int)v17) + 24LL);
      *a5 = result;
      *a6 = 1;
    }
    else
    {
      result = *(unsigned __int16 *)(*(_QWORD *)(*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD, __int64, __int64))(*a3 + 16LL))(
                                                  a3,
                                                  a3[1] - 40LL * (unsigned int)~v15,
                                                  (unsigned int)(*(_DWORD *)(a1 + 16) - 1),
                                                  a4,
                                                  a7)
                                   + 24LL);
      *a5 = result;
      *a6 = 1;
    }
  }
  else
  {
    v10 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)a2 + 568LL);
    if ( v10 == sub_2FE3130 )
      v11 = *(_QWORD *)(a2 + 8LL * (unsigned __int16)v8 + 3400);
    else
      v11 = v10(a2, *(_WORD *)(a1 + 24));
    *a5 = *(unsigned __int16 *)(*(_QWORD *)v11 + 24LL);
    v12 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)a2 + 576LL);
    if ( v12 == sub_2FE3140 )
      LOBYTE(result) = *(_BYTE *)(a2 + v8 + 5592);
    else
      LOBYTE(result) = v12(a2, v8);
    result = (unsigned __int8)result;
    *a6 = (unsigned __int8)result;
  }
  return result;
}
