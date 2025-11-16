// Function: sub_1F3CF70
// Address: 0x1f3cf70
//
__int64 __fastcall sub_1F3CF70(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int8 *v3; // rdx
  __int64 (*v4)(); // rax

  v3 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v4 = *(__int64 (**)())(*(_QWORD *)a1 + 824LL);
  if ( v4 == sub_1D12E00 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v4)(a1, *v3, *((_QWORD *)v3 + 1));
}
