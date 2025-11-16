// Function: sub_2226880
// Address: 0x2226880
//
_QWORD *__fastcall sub_2226880(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        int a5,
        char a6,
        __int64 a7,
        _DWORD *a8,
        _QWORD *a9)
{
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // rbx
  __int64 v16; // [rsp+0h] [rbp-80h]
  _BYTE *v18; // [rsp+30h] [rbp-50h] BYREF
  __int64 v19; // [rsp+38h] [rbp-48h]
  _BYTE v20[64]; // [rsp+40h] [rbp-40h] BYREF

  v18 = v20;
  v16 = sub_2243120(a7 + 208);
  v19 = 0;
  v20[0] = 0;
  if ( a6 )
    v12 = sub_2224080(a1, a2, a3, a4, a5, a7, a8, (__int64)&v18);
  else
    v12 = sub_2225410(a1, a2, a3, a4, a5, a7, a8, (__int64)&v18);
  v13 = v12;
  v14 = v19;
  if ( v19 )
  {
    sub_2251BE0(a9, v19, 0);
    (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, _QWORD))(*(_QWORD *)v16 + 88LL))(v16, v18, &v18[v14], *a9);
  }
  if ( v18 != v20 )
    j___libc_free_0((unsigned __int64)v18);
  return v13;
}
