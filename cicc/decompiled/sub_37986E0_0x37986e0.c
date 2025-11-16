// Function: sub_37986E0
// Address: 0x37986e0
//
unsigned __int8 *__fastcall sub_37986E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // r9
  __int128 *v6; // rdx
  unsigned __int8 *v7; // r12
  __int128 v9; // [rsp-10h] [rbp-40h]
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+8h] [rbp-28h]

  v2 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v3 = *(_QWORD **)(a1 + 8);
  v11 = 0;
  v5 = v4;
  v6 = *(__int128 **)(a2 + 40);
  v10 = 0;
  *((_QWORD *)&v9 + 1) = v5;
  *(_QWORD *)&v9 = v2;
  v7 = sub_3406EB0(v3, 0x170u, (__int64)&v10, 1, 0, v5, *v6, v9);
  if ( v10 )
    sub_B91220((__int64)&v10, v10);
  return v7;
}
