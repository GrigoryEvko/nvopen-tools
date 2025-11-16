// Function: sub_E9A820
// Address: 0xe9a820
//
__int64 __fastcall sub_E9A820(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  void *v6; // rax

  v5 = sub_E92830(a2, a1[1], a3, a4, a5);
  v6 = *(void **)v5;
  if ( *(_QWORD *)v5
    || (*(_BYTE *)(v5 + 9) & 0x70) == 0x20
    && *(char *)(v5 + 8) >= 0
    && (*(_BYTE *)(v5 + 8) |= 8u, v6 = sub_E807D0(*(_QWORD *)(v5 + 24)), (*(_QWORD *)v5 = v6) != 0) )
  {
    if ( v6 != off_4C5D170 )
      return v5;
  }
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, a2, 0);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v5, 0);
  return v5;
}
