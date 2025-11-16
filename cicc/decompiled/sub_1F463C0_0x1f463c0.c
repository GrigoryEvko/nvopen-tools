// Function: sub_1F463C0
// Address: 0x1f463c0
//
__int64 __fastcall sub_1F463C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r12
  __int64 (__fastcall *v5)(__int64, __int64, _QWORD); // rbx
  __int64 v6; // rax
  __int64 v7; // rax

  result = *(_QWORD *)(a1 + 208);
  if ( (*(_BYTE *)(result + 792) & 1) != 0 )
  {
    v4 = *(_QWORD *)(a1 + 160);
    v5 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v4 + 16LL);
    v6 = sub_16BA580(a1, a2, a3);
    v7 = sub_1E128C0(v6, a2);
    return v5(v4, v7, 0);
  }
  return result;
}
