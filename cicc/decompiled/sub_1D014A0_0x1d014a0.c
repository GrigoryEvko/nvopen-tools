// Function: sub_1D014A0
// Address: 0x1d014a0
//
__int64 __fastcall sub_1D014A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // r8
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // [rsp+8h] [rbp-8h] BYREF

  v8 = a2;
  v2 = sub_1D010E0(*(_QWORD **)(a1 + 16), *(_QWORD *)(a1 + 24), &v8);
  v4 = *(_QWORD *)(v3 + 24);
  v5 = v4 - 8;
  if ( v2 != (__int64 *)(v4 - 8) )
  {
    v6 = *v2;
    *v2 = *(_QWORD *)(v4 - 8);
    *(_QWORD *)(v4 - 8) = v6;
    v5 = *(_QWORD *)(v3 + 24) - 8LL;
  }
  result = v8;
  *(_QWORD *)(v3 + 24) = v5;
  *(_DWORD *)(result + 196) = 0;
  return result;
}
