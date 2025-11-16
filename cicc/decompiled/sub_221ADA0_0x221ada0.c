// Function: sub_221ADA0
// Address: 0x221ada0
//
_QWORD *__fastcall sub_221ADA0(
        __int64 a1,
        _QWORD *a2,
        int a3,
        _QWORD *a4,
        int a5,
        char a6,
        unsigned __int64 a7,
        _DWORD *a8,
        void **a9)
{
  _BYTE *v10; // r13
  _QWORD *v11; // rax
  size_t v12; // r14
  _QWORD *v13; // r15
  char v15; // al
  _BYTE *(__fastcall *v16)(__int64, _BYTE *, _BYTE *, void *); // rax
  void *desta; // [rsp+8h] [rbp-78h]
  _BYTE *v20; // [rsp+10h] [rbp-70h]
  _BYTE *srca; // [rsp+18h] [rbp-68h]
  _BYTE *v23; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _BYTE v25[64]; // [rsp+40h] [rbp-40h] BYREF

  v23 = v25;
  v10 = (_BYTE *)sub_222F790(a7 + 208);
  v25[0] = 0;
  n = 0;
  if ( a6 )
    v11 = sub_22185C0(a1, a2, a3, a4, a5, a7, a8, (__int64)&v23);
  else
    v11 = sub_2219940(a1, a2, a3, a4, a5, a7, a8, (__int64)&v23);
  v12 = n;
  v13 = v11;
  if ( !n )
    goto LABEL_4;
  sub_22410F0(a9, n, 0);
  desta = *a9;
  srca = v23;
  v20 = &v23[v12];
  v15 = v10[56];
  if ( v15 != 1 )
  {
    if ( !v15 )
      sub_2216D60((__int64)v10);
    v16 = *(_BYTE *(__fastcall **)(__int64, _BYTE *, _BYTE *, void *))(*(_QWORD *)v10 + 56LL);
    if ( v16 == sub_2216D40 )
    {
      if ( v20 != srca )
      {
LABEL_14:
        memcpy(desta, srca, v12);
        srca = v23;
        goto LABEL_5;
      }
    }
    else
    {
      v16((__int64)v10, srca, v20, desta);
    }
LABEL_4:
    srca = v23;
    goto LABEL_5;
  }
  if ( v20 != v23 )
    goto LABEL_14;
LABEL_5:
  if ( srca != v25 )
    j___libc_free_0((unsigned __int64)srca);
  return v13;
}
