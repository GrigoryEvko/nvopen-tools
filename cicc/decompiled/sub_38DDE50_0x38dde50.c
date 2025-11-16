// Function: sub_38DDE50
// Address: 0x38dde50
//
__int64 __fastcall sub_38DDE50(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  void *v3; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax

  v2 = sub_38D7790(a2, a1[1]);
  v3 = (void *)(*(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v3 )
  {
    if ( (*(_BYTE *)(v2 + 9) & 0xC) != 8 )
      goto LABEL_3;
    *(_BYTE *)(v2 + 8) |= 4u;
    v5 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v2 + 24));
    v6 = v5 | *(_QWORD *)v2 & 7LL;
    *(_QWORD *)v2 = v6;
    if ( !v5 )
      goto LABEL_3;
    v3 = (void *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !v3 )
    {
      v3 = 0;
      if ( (*(_BYTE *)(v2 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v2 + 8) |= 4u;
        v7 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v2 + 24));
        *(_QWORD *)v2 = v7 | *(_QWORD *)v2 & 7LL;
        if ( off_4CF6DB8 != (_UNKNOWN *)v7 )
          return v2;
LABEL_3:
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 160))(a1, a2, 0);
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v2, 0);
        return v2;
      }
    }
  }
  if ( off_4CF6DB8 == v3 )
    goto LABEL_3;
  return v2;
}
