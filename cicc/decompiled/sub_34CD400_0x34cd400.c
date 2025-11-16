// Function: sub_34CD400
// Address: 0x34cd400
//
__int64 __fastcall sub_34CD400(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v7; // r12

  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
  sub_2FF0740(v7, (_BYTE *)(v7 + 273), a3);
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL))(a2, v7, 1);
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL))(a2, a4, 1);
  if ( (unsigned __int8)sub_2FF3FD0((_QWORD *)v7) )
    return 0;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 248LL))(v7);
  *(_BYTE *)(v7 + 272) = 1;
  return v7;
}
