// Function: sub_3574BC0
// Address: 0x3574bc0
//
__int64 __fastcall sub_3574BC0(_QWORD *a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  _QWORD *v4; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 (*v9)(); // r10
  __int64 result; // rax
  _QWORD *v11; // [rsp+0h] [rbp-20h] BYREF
  _QWORD *v12; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(__int64 (**)())(**(_QWORD **)(a1[27] + 16LL) + 128LL);
  if ( v3 == sub_2DAC790 )
    BUG();
  v4 = (_QWORD *)v3();
  v5 = a1[73];
  v6 = v4;
  v7 = *v4;
  v8 = a1[28];
  v12 = a1;
  v11 = a1;
  v9 = *(__int64 (**)())(v7 + 1584);
  result = 0;
  if ( v9 != sub_2FDC8B0 )
    return ((__int64 (__fastcall *)(_QWORD *, __int64 (__fastcall *)(), _QWORD **, __int64 (__fastcall *)(__int64, int), _QWORD **, __int64, __int64, __int64, _QWORD *))v9)(
             v6,
             sub_3578EF0,
             &v11,
             sub_3574310,
             &v12,
             v8,
             v5,
             a2,
             a1 + 193);
  return result;
}
