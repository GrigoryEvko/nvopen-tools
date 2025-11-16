// Function: sub_24E80F0
// Address: 0x24e80f0
//
_QWORD *__fastcall sub_24E80F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _BYTE *v4; // rbx
  _BYTE *v5; // r12
  void (__fastcall *v6)(_BYTE *, _BYTE *, __int64); // rax
  _QWORD v8[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 (__fastcall *v9)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-60h]
  __int64 (__fastcall *v10)(__int64 (__fastcall **)(__int64), __int64); // [rsp+18h] [rbp-58h]
  _BYTE *v11; // [rsp+20h] [rbp-50h] BYREF
  __int64 v12; // [rsp+28h] [rbp-48h]
  _BYTE v13[64]; // [rsp+30h] [rbp-40h] BYREF

  v12 = 0x100000000LL;
  v11 = v13;
  v8[0] = sub_24F52F0;
  v10 = sub_24E3D20;
  v9 = sub_24E3D30;
  sub_24E7E90(a1, a3, a4, (__int64)v8, (__int64)&v11);
  if ( v9 )
    v9(v8, v8, 3);
  v4 = v11;
  v5 = &v11[32 * (unsigned int)v12];
  if ( v11 != v5 )
  {
    do
    {
      v6 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v5 - 2);
      v5 -= 32;
      if ( v6 )
        v6(v5, v5, 3);
    }
    while ( v4 != v5 );
    v5 = v11;
  }
  if ( v5 != v13 )
    _libc_free((unsigned __int64)v5);
  (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*a1 + 16LL))(*a1);
  return a1;
}
