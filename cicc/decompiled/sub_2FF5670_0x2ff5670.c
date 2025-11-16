// Function: sub_2FF5670
// Address: 0x2ff5670
//
bool __fastcall sub_2FF5670(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 (*v5)(void); // rax
  unsigned __int64 v6; // rax

  v3 = 0;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  if ( v5 != sub_2DAC790 )
    v3 = v5();
  v6 = sub_2EBEE90(*(_QWORD *)(a2 + 32), *(_DWORD *)(a3 + 112));
  return !v6
      || (*(_WORD *)(v6 + 68) != 10 || (*(_DWORD *)(v6 + 40) & 0xFFFFFF) != 1)
      && ((*(_BYTE *)(*(_QWORD *)(v6 + 16) + 27LL) & 0x20) == 0
       || !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v3 + 56LL))(v3))
      || *(_DWORD *)(a3 + 8) <= (unsigned int)qword_502A1A8;
}
