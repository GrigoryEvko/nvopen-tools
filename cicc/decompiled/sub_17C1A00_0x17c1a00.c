// Function: sub_17C1A00
// Address: 0x17c1a00
//
__int64 __fastcall sub_17C1A00(__int64 a1, _QWORD *a2, double a3, double a4, double a5)
{
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // zf

  v5 = *(__int64 **)(a1 + 8);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F9B6E8 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_10;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F9B6E8);
  *(_QWORD *)(a1 + 208) = a2;
  v9 = *(_BYTE *)(a1 + 160) == 0;
  *(_QWORD *)(a1 + 216) = v8 + 360;
  *(_QWORD *)(a1 + 224) = *a2;
  if ( !v9 )
    sub_17BF860(a1 + 160);
  if ( *(_BYTE *)(a1 + 161) )
    return sub_17BD2C0(a1 + 160, a3, a4, a5);
  else
    return 0;
}
