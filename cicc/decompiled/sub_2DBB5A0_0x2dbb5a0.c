// Function: sub_2DBB5A0
// Address: 0x2dbb5a0
//
__int64 __fastcall sub_2DBB5A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 v10; // rax

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_10;
  }
  v5 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_5027190)
                 + 256);
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL);
  if ( v6 == sub_23CE270 )
    BUG();
  v7 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a2);
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 144LL);
  if ( v8 == sub_2C8F680 )
    return sub_2DBA470(a2, 0);
  v10 = ((__int64 (__fastcall *)(__int64, _QWORD))v8)(v7, 0);
  return sub_2DBA470(a2, v10);
}
