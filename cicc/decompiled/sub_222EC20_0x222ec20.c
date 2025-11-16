// Function: sub_222EC20
// Address: 0x222ec20
//
__int64 __fastcall sub_222EC20(__int64 a1, unsigned int a2)
{
  __int64 (__fastcall *v2)(__int64, unsigned int); // rax

  sub_2216D60(a1);
  v2 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)a1 + 48LL);
  if ( v2 == sub_CE72A0 )
    return a2;
  else
    return v2(a1, (char)a2);
}
