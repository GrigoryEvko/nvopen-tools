// Function: sub_E9A100
// Address: 0xe9a100
//
__int64 __fastcall sub_E9A100(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  void (*v5)(); // rcx
  __int64 (__fastcall *v6)(__int64 *, __int64, __int64); // rcx
  char v7; // al
  __int64 v8; // rdx

  sub_E9A030(a1);
  v4 = *a1;
  v5 = *(void (**)())(*a1 + 120);
  if ( v5 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, __int64, __int64))v5)(a1, a3, 1);
    v4 = *a1;
  }
  v6 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64))(v4 + 536);
  v7 = *(_BYTE *)(a1[1] + 1906);
  if ( v7 )
  {
    if ( v7 != 1 )
      BUG();
    v8 = 8;
  }
  else
  {
    v8 = 4;
  }
  return v6(a1, a2, v8);
}
