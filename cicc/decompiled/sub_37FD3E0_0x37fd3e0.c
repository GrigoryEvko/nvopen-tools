// Function: sub_37FD3E0
// Address: 0x37fd3e0
//
_QWORD *__fastcall sub_37FD3E0(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  unsigned int v7; // ecx
  _QWORD *v8; // r12
  __int64 v10; // rdx
  __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  int v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+10h] [rbp-30h]

  v2 = (_QWORD *)a1[1];
  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v11, *a1, v2[8], v5, *((_QWORD *)v4 + 1));
    v6 = v13;
    v7 = (unsigned __int16)v12;
  }
  else
  {
    v7 = v3(*a1, v2[8], v5, *((_QWORD *)v4 + 1));
    v6 = v10;
  }
  v11 = 0;
  v12 = 0;
  v8 = sub_33F17F0(v2, 51, (__int64)&v11, v7, v6);
  if ( v11 )
    sub_B91220((__int64)&v11, v11);
  return v8;
}
