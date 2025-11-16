// Function: sub_222F5F0
// Address: 0x222f5f0
//
__int64 __fastcall sub_222F5F0(_BYTE *a1, _BYTE *a2, _BYTE *a3, void *a4)
{
  char v6; // al
  _BYTE *(__fastcall *v7)(__int64, _BYTE *, _BYTE *, void *); // rax

  v6 = a1[56];
  if ( v6 != 1 )
  {
    if ( !v6 )
      sub_2216D60((__int64)a1);
    v7 = *(_BYTE *(__fastcall **)(__int64, _BYTE *, _BYTE *, void *))(*(_QWORD *)a1 + 56LL);
    if ( v7 != sub_2216D40 )
      return (__int64)v7((__int64)a1, a2, a3, a4);
  }
  if ( a2 == a3 )
    return (__int64)a2;
  memcpy(a4, a2, a3 - a2);
  return (__int64)a3;
}
