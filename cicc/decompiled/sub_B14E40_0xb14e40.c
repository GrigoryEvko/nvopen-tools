// Function: sub_B14E40
// Address: 0xb14e40
//
__int64 __fastcall sub_B14E40(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 (__fastcall *v3)(__int64, _QWORD *); // r15
  __int64 v4; // rax
  __int64 (__fastcall *v5)(__int64, char *); // rax
  __int64 result; // rax
  __int64 v7; // rax
  _QWORD v8[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v9; // [rsp+10h] [rbp-40h] BYREF

  v2 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL))(a2, "call to ");
  v3 = *(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 56LL);
  sub_E0CEC0(v8, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 16));
  v4 = v3(v2, v8);
  (*(void (__fastcall **)(__int64, const char *))(*(_QWORD *)v4 + 48LL))(v4, " marked \"dontcall-");
  if ( (__int64 *)v8[0] != &v9 )
    j_j___libc_free_0(v8[0], v9 + 1);
  v5 = *(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL);
  if ( !*(_BYTE *)(a1 + 12) )
  {
    result = v5(a2, "error\"");
    if ( !*(_QWORD *)(a1 + 40) )
      return result;
    goto LABEL_5;
  }
  result = v5(a2, "warn\"");
  if ( *(_QWORD *)(a1 + 40) )
  {
LABEL_5:
    v7 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL))(a2, ": ");
    return (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v7 + 40LL))(
             v7,
             *(_QWORD *)(a1 + 32),
             *(_QWORD *)(a1 + 40));
  }
  return result;
}
