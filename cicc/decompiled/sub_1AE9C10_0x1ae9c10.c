// Function: sub_1AE9C10
// Address: 0x1ae9c10
//
char __fastcall sub_1AE9C10(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // r14
  char result; // al
  __int64 v8; // rax
  __int64 v9; // rax

  v4 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 24 * (1 - v4)) + 24LL);
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 24 * (2 - v4)) + 24LL);
  result = sub_1AE8290(v5, v6, (__int64)a2);
  if ( !result )
  {
    result = sub_1AE93B0(*a2, a1);
    if ( result )
    {
      v8 = sub_15C70A0(a1 + 48);
      v9 = sub_15A76D0(a3, (__int64)a2, v5, v6, v8, 0);
      return sub_15F2180(v9, (__int64)a2);
    }
  }
  return result;
}
