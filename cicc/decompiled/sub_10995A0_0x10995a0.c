// Function: sub_10995A0
// Address: 0x10995a0
//
__int64 *__fastcall sub_10995A0(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  bool v6; // zf
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx
  _BYTE *v11; // [rsp+0h] [rbp-50h] BYREF
  __int16 v12; // [rsp+20h] [rbp-30h]

  v5 = a3;
  v6 = *a2 == 0;
  v12 = 257;
  if ( !v6 )
  {
    v11 = a2;
    LOBYTE(v12) = 3;
  }
  v7 = sub_2241E50(a1, a2, a3, a4, a5);
  v8 = sub_22077B0(64);
  v9 = v8;
  if ( v8 )
    sub_C63EB0(v8, (__int64)&v11, v5, v7);
  *a1 = v9 | 1;
  return a1;
}
