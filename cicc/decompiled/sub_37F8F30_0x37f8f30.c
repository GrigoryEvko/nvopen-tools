// Function: sub_37F8F30
// Address: 0x37f8f30
//
__int64 __fastcall sub_37F8F30(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 (*v5)(); // rdx
  __int64 v6; // rax

  *(_QWORD *)(a1 + 200) = a2;
  *(_QWORD *)(a1 + 208) = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 16LL);
  *(_QWORD *)(a1 + 208) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 200LL))(v2);
  v5 = *(__int64 (**)())(*(_QWORD *)v2 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
    v6 = ((__int64 (__fastcall *)(__int64))v5)(v2);
  *(_QWORD *)(a1 + 216) = v6;
  sub_37F6F10(a1, a2, (__int64)v5, v3, v4);
  sub_37F7FF0(a1);
  if ( (_BYTE)qword_50515A8 )
    sub_37F8050(a1, *(__int64 **)(a1 + 200));
  return 0;
}
