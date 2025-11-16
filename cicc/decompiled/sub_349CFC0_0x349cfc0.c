// Function: sub_349CFC0
// Address: 0x349cfc0
//
__int64 __fastcall sub_349CFC0(__int64 a1, __int64 a2)
{
  char v3; // r12
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 (__fastcall ***v6)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD); // rdi
  __int64 v7; // r12

  v3 = qword_503A428 | sub_2E799E0(a2);
  v4 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5027190);
  v5 = v4;
  if ( v4 )
    v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_5027190);
  *(_QWORD *)(a1 + 216) = v5;
  if ( v3 )
  {
    *(_QWORD *)(a1 + 328) = a2;
    v7 = a1 + 224;
    *(_DWORD *)(a1 + 344) = *(_DWORD *)(a2 + 120);
    sub_2E708A0(a1 + 224);
    v6 = *(__int64 (__fastcall ****)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(a1 + 200);
    v5 = *(_QWORD *)(a1 + 216);
  }
  else
  {
    v6 = *(__int64 (__fastcall ****)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(a1 + 208);
    v7 = 0;
  }
  return (**v6)(v6, a2, v7, v5, (unsigned int)qword_503A268, (unsigned int)qword_503A188);
}
