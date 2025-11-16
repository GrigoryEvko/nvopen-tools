// Function: sub_2FE3A30
// Address: 0x2fe3a30
//
__int64 __fastcall sub_2FE3A30(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int16 *v3; // rdx
  __int64 (*v4)(); // rax

  v3 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v4 = *(__int64 (**)())(*(_QWORD *)a1 + 1392LL);
  if ( v4 == sub_2FE3480 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v4)(a1, *v3, *((_QWORD *)v3 + 1));
}
