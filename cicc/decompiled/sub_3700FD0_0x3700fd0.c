// Function: sub_3700FD0
// Address: 0x3700fd0
//
__int64 *__fastcall sub_3700FD0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  unsigned __int8 v5; // al
  __int64 v7; // rdi

  v3 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)(v3 + 48) )
  {
    v4 = *(_QWORD *)(v3 + 40);
  }
  else
  {
    v7 = *(_QWORD *)(v3 + 24);
    v4 = 0;
    if ( v7 )
      v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL))(v7) - *(_QWORD *)(v3 + 32);
  }
  if ( *(_QWORD *)(v3 + 56) == v4 || (v5 = sub_1254BC0(*(_QWORD *)(a2 + 40)), v5 <= 0xEFu) )
  {
    *a1 = 1;
    return a1;
  }
  else
  {
    sub_1254B30(a1, *(_QWORD *)(a2 + 40), v5 & 0xF);
    return a1;
  }
}
