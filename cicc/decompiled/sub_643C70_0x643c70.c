// Function: sub_643C70
// Address: 0x643c70
//
__int64 __fastcall sub_643C70(__int64 a1)
{
  __int64 result; // rax

  result = qword_4CFDE78;
  qword_4CFDE78 = a1;
  *(_QWORD *)(a1 + 464) = result;
  return result;
}
