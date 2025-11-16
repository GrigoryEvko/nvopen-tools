// Function: sub_23CF320
// Address: 0x23cf320
//
void *__fastcall sub_23CF320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 (*v5)(); // rax
  __int64 v6; // rax

  if ( a5 || (*(_BYTE *)(a3 + 32) & 0xF) != 8 )
    return sub_E410F0(a4, a2, a3);
  v5 = *(__int64 (**)())(*(_QWORD *)a1 + 24LL);
  if ( v5 == sub_23CE280 )
    BUG();
  v6 = ((__int64 (__fastcall *)(__int64))v5)(a1);
  return (void *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v6 + 88LL))(v6, a2, a3, a1);
}
