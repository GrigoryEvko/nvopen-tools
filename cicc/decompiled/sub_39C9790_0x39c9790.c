// Function: sub_39C9790
// Address: 0x39c9790
//
void __fastcall sub_39C9790(__int64 *a1, __int64 a2, __int16 a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  unsigned __int64 *v13[2]; // [rsp+0h] [rbp-B0h] BYREF
  _BYTE v14[8]; // [rsp+10h] [rbp-A0h] BYREF
  char *v15; // [rsp+18h] [rbp-98h]
  char v16; // [rsp+28h] [rbp-88h] BYREF
  int v17; // [rsp+5Ch] [rbp-54h]
  __int64 **v18; // [rsp+78h] [rbp-38h]

  v8 = sub_145CDC0(0x10u, a1 + 11);
  if ( v8 )
  {
    *(_QWORD *)v8 = 0;
    *(_DWORD *)(v8 + 8) = 0;
  }
  sub_39A1E10((__int64)v14, a1[24], (__int64)a1, v8);
  if ( !*(_BYTE *)a4 )
    v17 = 2;
  v9 = a1[24];
  v10 = 0;
  v13[0] = 0;
  v13[1] = 0;
  v11 = *(_QWORD *)(*(_QWORD *)(v9 + 264) + 16LL);
  v12 = *(__int64 (**)())(*(_QWORD *)v11 + 112LL);
  if ( v12 != sub_1D00B10 )
    v10 = ((__int64 (__fastcall *)(__int64, _QWORD))v12)(v11, 0);
  if ( (unsigned __int8)sub_399F750((__int64)v14, v10, v13, *(_DWORD *)(a4 + 4)) )
  {
    sub_399FAC0(v14, v13, 0);
    sub_399FD30((__int64)v14);
    sub_39A4520(a1, a2, a3, v18);
  }
  if ( v15 != &v16 )
    _libc_free((unsigned __int64)v15);
}
