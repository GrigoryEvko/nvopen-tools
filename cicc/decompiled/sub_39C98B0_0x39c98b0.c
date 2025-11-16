// Function: sub_39C98B0
// Address: 0x39c98b0
//
void __fastcall sub_39C98B0(__int64 *a1, __int64 a2, __int64 a3, __int16 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  unsigned __int64 *v14; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v15; // [rsp+18h] [rbp-A8h]
  _BYTE v16[8]; // [rsp+20h] [rbp-A0h] BYREF
  char *v17; // [rsp+28h] [rbp-98h]
  char v18; // [rsp+38h] [rbp-88h] BYREF
  int v19; // [rsp+6Ch] [rbp-54h]
  __int64 **v20; // [rsp+88h] [rbp-38h]

  v8 = sub_145CDC0(0x10u, a1 + 11);
  if ( v8 )
  {
    *(_QWORD *)v8 = 0;
    *(_DWORD *)(v8 + 8) = 0;
  }
  sub_39A1E10((__int64)v16, a1[24], (__int64)a1, v8);
  if ( *(_DWORD *)(a2 + 48) )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
    sub_399FD50((__int64)v16, v9);
    if ( !*(_BYTE *)a5 )
      v19 = 2;
    v14 = 0;
    v15 = 0;
    if ( v9 )
    {
      v14 = *(unsigned __int64 **)(v9 + 24);
      v15 = *(_QWORD *)(v9 + 32);
    }
  }
  else
  {
    sub_399FD50((__int64)v16, 0);
    if ( !*(_BYTE *)a5 )
      v19 = 2;
    v14 = 0;
    v15 = 0;
  }
  v10 = 0;
  v11 = *(_QWORD *)(*(_QWORD *)(a1[24] + 264) + 16LL);
  v12 = *(__int64 (**)())(*(_QWORD *)v11 + 112LL);
  if ( v12 != sub_1D00B10 )
    v10 = ((__int64 (__fastcall *)(__int64, _QWORD))v12)(v11, 0);
  if ( (unsigned __int8)sub_399F750((__int64)v16, v10, &v14, *(_DWORD *)(a5 + 4)) )
  {
    sub_399FAC0(v16, &v14, 0);
    sub_399FD30((__int64)v16);
    sub_39A4520(a1, a3, a4, v20);
  }
  if ( v17 != &v18 )
    _libc_free((unsigned __int64)v17);
}
