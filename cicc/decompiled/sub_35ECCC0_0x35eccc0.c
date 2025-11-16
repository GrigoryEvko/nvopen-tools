// Function: sub_35ECCC0
// Address: 0x35eccc0
//
__int64 __fastcall sub_35ECCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 (*v5)(); // rax
  __int64 result; // rax

  v5 = *(__int64 (**)())(**(_QWORD **)(a1 + 40) + 280LL);
  if ( v5 == sub_3059470 )
    return sub_35EC3C0(a1, a2, a3, a4, a5);
  result = v5();
  if ( (_BYTE)result )
    return sub_35EC3C0(a1, a2, a3, a4, a5);
  return result;
}
