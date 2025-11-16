// Function: sub_39AC9D0
// Address: 0x39ac9d0
//
__int64 __fastcall sub_39AC9D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned int v17; // esi
  __int64 v18; // rax

  v3 = a3;
  v4 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 32) = a2;
  result = *(_QWORD *)(v4 + 264);
  v7 = *(_QWORD *)result;
  if ( !a3 )
  {
    v3 = sub_39AC850(a2);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 272LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      v3);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 280LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      3);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 288LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      32);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 296LL))(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL));
    v16 = *(_QWORD *)(a1 + 8);
    v17 = *(_DWORD *)(a2 + 176);
    v18 = *(_QWORD *)(v16 + 264);
    if ( *(_DWORD *)(v18 + 340) >= v17 )
      v17 = *(_DWORD *)(v18 + 340);
    sub_396F480(v16, v17, v7);
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
               *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
               v3,
               0);
  }
  if ( *(_BYTE *)(a1 + 26) || *(_BYTE *)(a1 + 24) )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v9 = 0;
    v10 = *(_QWORD *)(v8 + 256);
    v11 = *(unsigned int *)(v10 + 120);
    if ( (_DWORD)v11 )
      v9 = *(_QWORD *)(*(_QWORD *)(v10 + 112) + 32 * v11 - 32);
    *(_QWORD *)(a1 + 40) = v9;
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v8 + 256) + 872LL))(
               *(_QWORD *)(v8 + 256),
               v3,
               0);
    if ( *(_BYTE *)(a1 + 24) )
    {
      v12 = sub_396DD80(*(_QWORD *)(a1 + 8));
      if ( (*(_BYTE *)(v7 + 18) & 8) == 0 || (v15 = sub_15E38F0(v7), v13 = sub_1649C60(v15), *(_BYTE *)(v13 + 16)) )
        v13 = 0;
      v14 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v12 + 88LL))(
              v12,
              v13,
              *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL),
              *(_QWORD *)(a1 + 16));
      result = *(_QWORD *)(a1 + 32);
      if ( !*(_BYTE *)(result + 184) )
        return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8)
                                                                                                  + 256LL)
                                                                                    + 960LL))(
                 *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
                 v14,
                 1,
                 1,
                 0);
    }
  }
  return result;
}
