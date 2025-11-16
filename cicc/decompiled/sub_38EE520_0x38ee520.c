// Function: sub_38EE520
// Address: 0x38ee520
//
__int64 __fastcall sub_38EE520(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned int v4; // r12d
  void (__fastcall *v5)(__int64, const char *, __int64); // rax
  const char *v7; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a1;
  if ( !*(_BYTE *)(v2 + 845) )
  {
    if ( (unsigned __int8)sub_38E36C0(v2) )
      return 1;
    v2 = *a1;
  }
  if ( (unsigned __int8)sub_38EE320(v2, &v7, v8) )
    return 1;
  v3 = *(_QWORD *)(*a1 + 328);
  v4 = *(unsigned __int8 *)(*(_QWORD *)(*a1 + 336) + 16LL);
  v5 = *(void (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v3 + 424LL);
  if ( (_BYTE)v4 )
  {
    v5(v3, (const char *)v8[0], 8);
    (*(void (__fastcall **)(_QWORD, const char *, __int64))(**(_QWORD **)(*a1 + 328) + 424LL))(
      *(_QWORD *)(*a1 + 328),
      v7,
      8);
    return 0;
  }
  else
  {
    v5(v3, v7, 8);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(*a1 + 328) + 424LL))(
      *(_QWORD *)(*a1 + 328),
      v8[0],
      8);
    return v4;
  }
}
