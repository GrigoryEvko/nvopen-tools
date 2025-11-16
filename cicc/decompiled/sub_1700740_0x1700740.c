// Function: sub_1700740
// Address: 0x1700740
//
__int64 __fastcall sub_1700740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 (*v5)(); // rax
  __int64 v6; // rax

  if ( a5 || (*(_BYTE *)(a3 + 32) & 0xF) != 8 )
    return sub_38BA670(a4, a2, a3, 0);
  v5 = *(__int64 (**)())(*(_QWORD *)a1 + 24LL);
  if ( v5 == sub_16FF760 )
    BUG();
  v6 = ((__int64 (__fastcall *)(__int64))v5)(a1);
  return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v6 + 48LL))(v6, a2, a3, a1);
}
