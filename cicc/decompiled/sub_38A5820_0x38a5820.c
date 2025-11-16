// Function: sub_38A5820
// Address: 0x38a5820
//
__int64 __fastcall sub_38A5820(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // rbp
  __int64 v8; // rdi
  unsigned __int64 v9; // rsi
  const char *v10; // [rsp-28h] [rbp-28h] BYREF
  char v11; // [rsp-18h] [rbp-18h]
  char v12; // [rsp-17h] [rbp-17h]
  __int64 v13; // [rsp-8h] [rbp-8h]

  if ( a3 )
    return sub_38A4F40(a1, a2, a4, a5, a6);
  v13 = v6;
  v8 = a1 + 8;
  v9 = *(_QWORD *)(v8 + 48);
  v12 = 1;
  v10 = "missing 'distinct', required for !DICompileUnit";
  v11 = 3;
  return sub_38814C0(v8, v9, (__int64)&v10);
}
