// Function: sub_C637E0
// Address: 0xc637e0
//
_QWORD *__fastcall sub_C637E0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD v3[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v4[3]; // [rsp+10h] [rbp-20h] BYREF

  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 16) + 32LL))(
    v3,
    *(_QWORD *)(a1 + 16),
    *(unsigned int *)(a1 + 8));
  sub_CB6200(a2, v3[0], v3[1]);
  result = v4;
  if ( (_QWORD *)v3[0] != v4 )
    return (_QWORD *)j_j___libc_free_0(v3[0], v4[0] + 1LL);
  return result;
}
