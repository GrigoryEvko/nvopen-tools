// Function: sub_28023E0
// Address: 0x28023e0
//
__int64 __fastcall sub_28023E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9

  if ( (unsigned __int8)sub_27FF780(a3, *(_QWORD *)(a5 + 16), *(_QWORD *)(a5 + 24), *(__int64 **)(a5 + 32), a6) )
  {
    v8 = *(_QWORD *)(a5 + 16);
    sub_D50AF0(*(_QWORD *)(a5 + 24));
    sub_22D0390(a1, v8, v9, v10, v11, v12);
    return a1;
  }
  else
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    return a1;
  }
}
