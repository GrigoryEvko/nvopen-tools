// Function: sub_3258CC0
// Address: 0x3258cc0
//
__int64 __fastcall sub_3258CC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r12
  unsigned __int8 *v11; // rsi
  __int64 v12; // rsi
  unsigned __int8 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned __int8 v16; // [rsp+Eh] [rbp-22h]
  unsigned __int8 v17; // [rsp+Fh] [rbp-21h]

  v3 = a3;
  v5 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 32) = a2;
  result = *(_QWORD *)(v5 + 232);
  v7 = *(_QWORD *)result;
  if ( !a3 )
  {
    v3 = sub_3258B50(a2);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 312LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      v3);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 320LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      3);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 328LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      32);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 336LL))(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL));
    v14 = *(_QWORD *)(a1 + 8);
    v15 = *(_QWORD *)(v14 + 232);
    v17 = *(_BYTE *)(a2 + 208);
    v16 = *(_BYTE *)(v15 + 340);
    sub_31DCA70(v14, *(&v16 + (v17 > v16)), v7, 0);
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
               *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
               v3,
               0);
  }
  if ( *(_BYTE *)(a1 + 26) || *(_BYTE *)(a1 + 24) )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v9 = v3;
    *(_QWORD *)(a1 + 40) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 224) + 288LL) + 8LL);
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v8 + 224) + 1056LL))(
               *(_QWORD *)(v8 + 224),
               v3,
               0);
    if ( *(_BYTE *)(a1 + 24) )
    {
      v10 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
      if ( (*(_BYTE *)(v7 + 2) & 8) == 0 || (v13 = (unsigned __int8 *)sub_B2E500(v7), v11 = sub_BD3990(v13, v9), *v11) )
        v11 = 0;
      v12 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 *, _QWORD, _QWORD))(*(_QWORD *)v10 + 144LL))(
              v10,
              v11,
              *(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL),
              *(_QWORD *)(a1 + 16));
      result = *(_QWORD *)(a1 + 32);
      if ( !*(_BYTE *)(result + 236) )
        return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8)
                                                                                                  + 224LL)
                                                                                    + 1168LL))(
                 *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
                 v12,
                 1,
                 1,
                 0);
    }
  }
  return result;
}
