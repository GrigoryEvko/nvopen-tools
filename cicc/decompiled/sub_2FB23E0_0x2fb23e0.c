// Function: sub_2FB23E0
// Address: 0x2fb23e0
//
__int64 __fastcall sub_2FB23E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 *v8; // r8
  unsigned __int64 v9; // r10
  __int64 (__fastcall *v10)(__int64, __int64); // rax
  __int64 v11; // rax
  __int64 v13; // rax

  if ( !a2 )
    return 0;
  v4 = *(_QWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( !v4 )
    return 0;
  v6 = sub_2E8A250(a2, 0, *(_QWORD *)(a1 + 40), *(_QWORD **)(a1 + 48));
  if ( !v6 )
    return 0;
  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(__int64 **)(a1 + 48);
  v9 = *(_QWORD *)(*(_QWORD *)(v7 + 56)
                 + 16LL * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL) & 0x7FFFFFFF))
     & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(__int64 (__fastcall **)(__int64, __int64))(*v8 + 352);
  if ( v10 != sub_2EBDF80 )
  {
    v13 = ((__int64 (__fastcall *)(_QWORD, unsigned __int64, _QWORD))v10)(
            *(_QWORD *)(a1 + 48),
            *(_QWORD *)(*(_QWORD *)(v7 + 56)
                      + 16LL * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL) & 0x7FFFFFFF))
          & 0xFFFFFFFFFFFFFFF8LL,
            *(_QWORD *)(a3 + 32));
    v8 = *(__int64 **)(a1 + 48);
    v9 = v13;
  }
  v11 = sub_2E8A4A0(v4, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL), v9, *(_QWORD *)(a1 + 40), v8, 1);
  if ( v6 != v11 )
    return (*(_DWORD *)(*(_QWORD *)(v11 + 8) + 4 * ((unsigned __int64)*(unsigned __int16 *)(*(_QWORD *)v6 + 24LL) >> 5)) >> *(_WORD *)(*(_QWORD *)v6 + 24LL))
         & 1;
  else
    return 0;
}
