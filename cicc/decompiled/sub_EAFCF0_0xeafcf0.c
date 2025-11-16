// Function: sub_EAFCF0
// Address: 0xeafcf0
//
__int64 __fastcall sub_EAFCF0(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned int v4; // r12d
  void (__fastcall *v5)(__int64, const char *, __int64); // rax
  const char *v7; // [rsp+0h] [rbp-20h] BYREF
  const char *v8; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a1;
  if ( !*(_BYTE *)(v2 + 869) )
  {
    if ( (unsigned __int8)sub_EA2540(v2) )
      return 1;
    v2 = *a1;
  }
  if ( (unsigned __int8)sub_EAFAF0(v2, &v7, &v8) )
    return 1;
  v3 = *(_QWORD *)(*a1 + 232);
  v4 = *(unsigned __int8 *)(*(_QWORD *)(*a1 + 240) + 16LL);
  v5 = *(void (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v3 + 536LL);
  if ( (_BYTE)v4 )
  {
    v5(v3, v8, 8);
    (*(void (__fastcall **)(_QWORD, const char *, __int64))(**(_QWORD **)(*a1 + 232) + 536LL))(
      *(_QWORD *)(*a1 + 232),
      v7,
      8);
    return 0;
  }
  else
  {
    v5(v3, v7, 8);
    (*(void (__fastcall **)(_QWORD, const char *, __int64))(**(_QWORD **)(*a1 + 232) + 536LL))(
      *(_QWORD *)(*a1 + 232),
      v8,
      8);
    return v4;
  }
}
