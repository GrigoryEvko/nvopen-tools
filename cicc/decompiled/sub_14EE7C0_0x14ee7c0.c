// Function: sub_14EE7C0
// Address: 0x14ee7c0
//
__int64 __fastcall sub_14EE7C0(__int64 a1, _QWORD *a2, __int64 **a3, unsigned int a4, __int64 a5)
{
  __int64 *v9; // rdi
  unsigned __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r13
  _BYTE *v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // rdi
  __int64 v18; // rax
  char v19; // al
  char v20; // al
  __int64 v21; // r14
  __int64 v22; // rax
  _BYTE *v23; // [rsp+8h] [rbp-108h]
  __int64 v24; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v25[2]; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v26[2]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v27; // [rsp+40h] [rbp-D0h]
  void *s; // [rsp+50h] [rbp-C0h] BYREF
  size_t n; // [rsp+58h] [rbp-B8h]
  _BYTE v30[176]; // [rsp+60h] [rbp-B0h] BYREF

  v9 = *a3;
  v10 = *((unsigned int *)a3 + 2);
  s = v30;
  n = 0x8000000000LL;
  if ( (unsigned __int8)sub_14EA1E0((__int64)v9, v10, a4, (__int64)&s)
    || (v11 = a2[69], v12 = **a3, -1431655765 * (unsigned int)((a2[70] - v11) >> 3) <= (unsigned int)v12)
    || (v13 = *(_QWORD *)(v11 + 24LL * (unsigned int)v12 + 16)) == 0 )
  {
    v26[0] = "Invalid record";
    v27 = 259;
    sub_14EE4B0(v25, (__int64)(a2 + 1), (__int64)v26);
    v18 = v25[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v18 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    v25[1] = (unsigned int)n;
    v25[0] = (__int64)s;
    if ( (_DWORD)n )
    {
      v23 = s;
      v14 = memchr(s, 0, (unsigned int)n);
      if ( v14 )
      {
        if ( v14 - v23 != -1 )
        {
          v26[0] = "Invalid value name";
          v27 = 259;
          sub_14EE4B0(&v24, (__int64)(a2 + 1), (__int64)v26);
          v15 = v24;
          v16 = s;
          *(_BYTE *)(a1 + 8) |= 3u;
          *(_QWORD *)a1 = v15 & 0xFFFFFFFFFFFFFFFELL;
          if ( v16 != v30 )
            _libc_free((unsigned __int64)v16);
          return a1;
        }
      }
    }
    v27 = 261;
    v26[0] = v25;
    sub_164B780(v13, v26);
    v19 = *(_BYTE *)(v13 + 16);
    if ( (!v19 || v19 == 3) && *(_QWORD *)(v13 + 48) == 1 )
    {
      if ( *(_DWORD *)(a5 + 52) == 3 )
      {
        *(_QWORD *)(v13 + 48) = 0;
      }
      else
      {
        v21 = a2[55];
        v22 = sub_1649960(v13);
        *(_QWORD *)(v13 + 48) = sub_1633B90(v21, v22);
      }
    }
    v20 = *(_BYTE *)(a1 + 8);
    *(_QWORD *)a1 = v13;
    *(_BYTE *)(a1 + 8) = v20 & 0xFC | 2;
  }
  if ( s != v30 )
    _libc_free((unsigned __int64)s);
  return a1;
}
