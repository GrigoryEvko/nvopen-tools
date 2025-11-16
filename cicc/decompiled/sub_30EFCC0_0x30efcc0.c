// Function: sub_30EFCC0
// Address: 0x30efcc0
//
__int64 __fastcall sub_30EFCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax

  v6 = sub_B2BE50(a3);
  v7 = sub_B6F970(v6);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v7 + 40LL))(v7, "kernel-info", 11) )
    sub_30EDB30(a3, a4);
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
