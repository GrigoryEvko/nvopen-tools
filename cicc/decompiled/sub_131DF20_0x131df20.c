// Function: sub_131DF20
// Address: 0x131df20
//
__int64 __fastcall sub_131DF20(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 result; // rax

  sub_130B1D0((_QWORD *)a1, (__int64 *)a2);
  if ( (int)sub_130B150((_QWORD *)(a1 + 8), (_QWORD *)(a2 + 8)) < 0 )
    sub_130B140((__int64 *)(a1 + 8), (__int64 *)(a2 + 8));
  *(_QWORD *)(a1 + 16) += *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 24) += *(_QWORD *)(a2 + 24);
  v2 = *(_DWORD *)(a2 + 32);
  if ( *(_DWORD *)(a1 + 32) < v2 )
    *(_DWORD *)(a1 + 32) = v2;
  *(_DWORD *)(a1 + 36) += *(_DWORD *)(a2 + 36);
  *(_QWORD *)(a1 + 40) += *(_QWORD *)(a2 + 40);
  result = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 56) += result;
  return result;
}
