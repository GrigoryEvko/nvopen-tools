// Function: sub_378F060
// Address: 0x378f060
//
_QWORD *__fastcall sub_378F060(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v3; // rax
  unsigned __int16 v4; // si
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // r8
  unsigned int v8; // ecx
  _QWORD *v9; // rdi
  _QWORD *v10; // r12
  __int64 v12; // rdx
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  int v14; // [rsp+8h] [rbp-38h]
  __int64 v15; // [rsp+10h] [rbp-30h]

  v2 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v3 = *(__int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = a1[1];
  if ( v2 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v13, *a1, *(_QWORD *)(v6 + 64), v4, v5);
    v7 = v15;
    v8 = (unsigned __int16)v14;
  }
  else
  {
    v8 = v2(*a1, *(_QWORD *)(v6 + 64), v4, v5);
    v7 = v12;
  }
  v9 = (_QWORD *)a1[1];
  v13 = 0;
  v14 = 0;
  v10 = sub_33F17F0(v9, 51, (__int64)&v13, v8, v7);
  if ( v13 )
    sub_B91220((__int64)&v13, v13);
  return v10;
}
