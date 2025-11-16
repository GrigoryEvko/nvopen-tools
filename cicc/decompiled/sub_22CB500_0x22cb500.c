// Function: sub_22CB500
// Address: 0x22cb500
//
__int64 __fastcall sub_22CB500(__int64 a1, unsigned __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  unsigned __int64 v5; // rax
  void (__fastcall *v6)(unsigned __int8 **, unsigned __int8 **, __int64); // rax
  __int64 v8; // rdi
  unsigned __int8 v9; // al
  unsigned __int8 *v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+8h] [rbp-28h]
  void (__fastcall *v12)(unsigned __int8 **, unsigned __int8 **, __int64); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v13)(__int64, unsigned __int8 **, __int64, __int64 *); // [rsp+18h] [rbp-18h]

  v5 = *a3;
  if ( (unsigned __int8)v5 <= 0x36u && (v8 = 0x40540000000000LL, _bittest64(&v8, v5)) )
  {
    v9 = a3[1];
    v10 = a3;
    v11 = (v9 >> 1) & 3;
    v13 = sub_22BDB80;
    v6 = (void (__fastcall *)(unsigned __int8 **, unsigned __int8 **, __int64))sub_22BDAB0;
  }
  else
  {
    v10 = a3;
    v13 = sub_22BDA00;
    v6 = (void (__fastcall *)(unsigned __int8 **, unsigned __int8 **, __int64))sub_22BDAE0;
  }
  v12 = v6;
  sub_22CB1D0(a1, a2, (__int64)a3, a4, (unsigned __int64)&v10);
  if ( v12 )
    v12(&v10, &v10, 3);
  return a1;
}
