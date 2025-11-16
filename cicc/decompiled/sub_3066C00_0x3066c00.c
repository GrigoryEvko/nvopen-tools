// Function: sub_3066C00
// Address: 0x3066c00
//
unsigned __int64 __fastcall sub_3066C00(__int64 a1, unsigned int a2, __int64 a3, __int64 *a4)
{
  __int64 *v4; // rbx
  signed __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // eax
  bool v9; // of
  unsigned __int64 result; // rax
  unsigned int v11; // [rsp+14h] [rbp-24h]

  v4 = a4;
  v5 = sub_3065900(a1 + 8, a2, a3, a4[3], 0, 0, 0);
  if ( (unsigned int)*((unsigned __int8 *)v4 + 8) - 17 <= 1 )
    v4 = *(__int64 **)v4[2];
  v6 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), v4, 0);
  BYTE2(v11) = 0;
  v8 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 32) + 736LL))(
         *(_QWORD *)(a1 + 32),
         *v4,
         v6,
         v7,
         v11);
  v9 = __OFADD__(v5, v8);
  result = v5 + v8;
  if ( v9 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v5 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
