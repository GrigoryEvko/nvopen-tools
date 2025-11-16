// Function: sub_266EF60
// Address: 0x266ef60
//
__int64 __fastcall sub_266EF60(__int64 *a1, __int64 a2, __int64 *a3, _BYTE *a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  bool (__fastcall *v7)(__int64); // rdx

  v5 = *a1;
  v6 = *a3;
  v7 = *(bool (__fastcall **)(__int64))(*(_QWORD *)(*a1 + 88) + 24LL);
  if ( v7 != sub_2534ED0 )
  {
    if ( v7(v5 + 88) )
      goto LABEL_5;
    *a4 = 1;
    if ( !v6 )
      goto LABEL_5;
    goto LABEL_4;
  }
  if ( *(_BYTE *)(v5 + 97) != *(_BYTE *)(v5 + 96) )
  {
    *a4 = 1;
    if ( !v6 )
    {
LABEL_5:
      v5 = *a1;
      return *(_QWORD *)(v5 + 104);
    }
LABEL_4:
    sub_250ED80(a1[1], *a1, v6, 1);
    goto LABEL_5;
  }
  return *(_QWORD *)(v5 + 104);
}
