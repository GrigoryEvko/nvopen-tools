// Function: sub_1F3B7F0
// Address: 0x1f3b7f0
//
__int64 __fastcall sub_1F3B7F0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 (*v3)(); // rax

  v3 = *(__int64 (**)())(*(_QWORD *)a1 + 456LL);
  if ( v3 == sub_1F39460 )
    return 0;
  if ( (unsigned __int8)v3()
    && (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 464LL))(
         a1,
         a2,
         *(_QWORD *)(a2 + 24)) )
  {
    return sub_1F3B710(a1, a2, a3);
  }
  return 0;
}
