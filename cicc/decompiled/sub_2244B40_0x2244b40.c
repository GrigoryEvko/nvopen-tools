// Function: sub_2244B40
// Address: 0x2244b40
//
__int64 __fastcall sub_2244B40(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6, char a7, char a8)
{
  _QWORD *v9; // r13
  __int64 v11; // rbp
  __int64 v12; // r13
  __int64 v13; // rbp
  int v15; // [rsp+10h] [rbp-248h] BYREF
  int v16; // [rsp+14h] [rbp-244h]
  int v17; // [rsp+18h] [rbp-240h]
  int v18; // [rsp+1Ch] [rbp-23Ch]
  wchar_t s[142]; // [rsp+20h] [rbp-238h] BYREF

  v9 = (_QWORD *)(a4 + 208);
  v11 = sub_2243120((_QWORD *)(a4 + 208), a2);
  v12 = sub_2244AF0(v9, a2);
  v15 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 80LL))(v11, 37);
  if ( a8 )
  {
    v17 = a7;
    v16 = a8;
    v18 = 0;
  }
  else
  {
    v16 = a7;
    v17 = 0;
  }
  sub_2256790(v12, s, 128, &v15, a6);
  v13 = (int)wcslen(s);
  if ( !a3 )
    (*(__int64 (__fastcall **)(__int64, wchar_t *, __int64))(*(_QWORD *)a2 + 96LL))(a2, s, v13);
  return a2;
}
