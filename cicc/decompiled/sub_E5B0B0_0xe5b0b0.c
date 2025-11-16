// Function: sub_E5B0B0
// Address: 0xe5b0b0
//
__int64 __fastcall sub_E5B0B0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 (*v5)(); // rax
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rdi
  _WORD *v9; // rdx

  if ( *a3 != 4
    || (v5 = *(__int64 (**)())(*((_QWORD *)a3 - 1) + 56LL), v5 == sub_E4C910)
    || !((unsigned __int8 (__fastcall *)(_BYTE *))v5)(a3 - 8) )
  {
    v6 = *(_QWORD *)(a1 + 304);
    v7 = *(_QWORD *)(v6 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v6 + 24) - v7) <= 4 )
    {
      sub_CB6200(v6, ".set ", 5u);
    }
    else
    {
      *(_DWORD *)v7 = 1952805678;
      *(_BYTE *)(v7 + 4) = 32;
      *(_QWORD *)(v6 + 32) += 5LL;
    }
    sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
    v8 = *(_QWORD *)(a1 + 304);
    v9 = *(_WORD **)(v8 + 32);
    if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 1u )
    {
      sub_CB6200(v8, (unsigned __int8 *)", ", 2u);
    }
    else
    {
      *v9 = 8236;
      *(_QWORD *)(v8 + 32) += 2LL;
    }
    sub_E7FAD0(a3, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312), 0);
    sub_E4D880(a1);
  }
  return sub_E9A490(a1, a2, a3);
}
