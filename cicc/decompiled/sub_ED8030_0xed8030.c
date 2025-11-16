// Function: sub_ED8030
// Address: 0xed8030
//
_QWORD *__fastcall sub_ED8030(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD v5[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v6[4]; // [rsp+10h] [rbp-20h] BYREF

  *(_DWORD *)(a2 + 8) = 0;
  v5[0] = v6;
  v5[1] = 0;
  LOBYTE(v6[0]) = 0;
  sub_2240AE0(a2 + 16, v5);
  v3 = (_QWORD *)v5[0];
  *a1 = 1;
  if ( v3 != v6 )
    j_j___libc_free_0(v3, v6[0] + 1LL);
  return a1;
}
