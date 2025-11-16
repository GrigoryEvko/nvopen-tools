// Function: sub_2EBE5D0
// Address: 0x2ebe5d0
//
__int64 __fastcall sub_2EBE5D0(_QWORD *a1, int a2)
{
  __int64 v3; // r12
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 *v9; // r15
  __int64 (__fastcall *v10)(__int64, __int64); // rax
  unsigned __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax

  v3 = 0;
  v4 = *(__int64 **)(*a1 + 16LL);
  v5 = *v4;
  v6 = *(__int64 (**)(void))(*v4 + 128);
  if ( v6 != sub_2DAC790 )
  {
    v3 = v6();
    v5 = **(_QWORD **)(*a1 + 16LL);
  }
  v7 = 16LL * (a2 & 0x7FFFFFFF);
  v8 = *(_QWORD *)(a1[7] + v7);
  v9 = (__int64 *)(*(__int64 (**)(void))(v5 + 200))();
  v10 = *(__int64 (__fastcall **)(__int64, __int64))(*v9 + 352);
  if ( v10 == sub_2EBDF80 )
    return 0;
  v12 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = ((__int64 (__fastcall *)(__int64 *, unsigned __int64, _QWORD))v10)(v9, v12, *a1);
  if ( v12 == v13 )
    return 0;
  if ( a2 < 0 )
    v14 = *(_QWORD *)(a1[7] + v7 + 8);
  else
    v14 = *(_QWORD *)(a1[38] + 8LL * (unsigned int)a2);
  if ( v14 )
  {
    if ( (*(_BYTE *)(v14 + 4) & 8) != 0 )
    {
      while ( 1 )
      {
        v14 = *(_QWORD *)(v14 + 32);
        if ( !v14 )
          break;
        if ( (*(_BYTE *)(v14 + 4) & 8) == 0 )
          goto LABEL_10;
      }
    }
    else
    {
LABEL_10:
      v15 = sub_2E8A3A0(
              *(_QWORD *)(v14 + 16),
              -858993459 * (unsigned int)((v14 - *(_QWORD *)(*(_QWORD *)(v14 + 16) + 32LL)) >> 3),
              v13,
              v3,
              v9);
      v13 = v15;
      if ( !v15 || v15 == v12 )
        return 0;
      while ( 1 )
      {
        v14 = *(_QWORD *)(v14 + 32);
        if ( !v14 )
          break;
        if ( (*(_BYTE *)(v14 + 4) & 8) == 0 )
          goto LABEL_10;
      }
    }
  }
  sub_2EBE4E0((__int64)a1, a2, v13);
  return 1;
}
