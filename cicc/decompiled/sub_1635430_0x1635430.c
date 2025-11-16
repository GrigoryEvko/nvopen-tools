// Function: sub_1635430
// Address: 0x1635430
//
__int64 __fastcall sub_1635430(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  size_t v4; // r14
  const char *v5; // r15
  const char *v6; // rax
  size_t v7; // rdx
  __int64 v8[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v9[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = 1;
  if ( *(_BYTE *)(a1 + 8) )
  {
    v8[0] = (__int64)v9;
    sub_1634F50(v8, "loop", (__int64)"");
    v4 = v8[1];
    v5 = (const char *)v8[0];
    v6 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
    v2 = sub_1635030(a1, v6, v7, v5, v4);
    if ( (_QWORD *)v8[0] != v9 )
      j_j___libc_free_0(v8[0], v9[0] + 1LL);
  }
  return v2;
}
