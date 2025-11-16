// Function: sub_398A6E0
// Address: 0x398a6e0
//
__int64 __fastcall sub_398A6E0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  void (__fastcall *v5)(__int64, _QWORD, _QWORD); // rbx
  __int64 v6; // rax

  result = *(unsigned int *)(a1 + 4216);
  if ( (_DWORD)result )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(_QWORD *)(v3 + 256);
    v5 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v4 + 160LL);
    v6 = sub_396DD80(v3);
    v5(v4, *(_QWORD *)(v6 + 184), 0);
    return sub_39BDF60(*(_QWORD *)(a1 + 8), a1 + 5552, a1, *(_QWORD *)(a1 + 4208), *(unsigned int *)(a1 + 4216));
  }
  return result;
}
