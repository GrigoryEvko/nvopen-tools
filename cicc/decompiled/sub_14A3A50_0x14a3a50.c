// Function: sub_14A3A50
// Address: 0x14a3a50
//
__int64 __fastcall sub_14A3A50(__int64 *a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // r8

  v4 = *a1;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 856LL);
  if ( v5 == sub_14A0B80 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64))v5)(
             v4,
             a2,
             a3,
             (unsigned __int16)a4 | ((unsigned __int64)BYTE2(a4) << 16));
}
