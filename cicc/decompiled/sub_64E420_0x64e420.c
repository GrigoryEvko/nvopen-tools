// Function: sub_64E420
// Address: 0x64e420
//
__int64 __fastcall sub_64E420(__int64 a1, __int64 a2, unsigned int *a3)
{
  char v5; // al
  __int64 v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v10; // rax

  v5 = *(_BYTE *)(a1 + 80);
  if ( v5 == 9 || v5 == 7 )
  {
    v6 = *(_QWORD *)(a1 + 88);
    v7 = *(_QWORD *)(v6 + 120);
    if ( v7 == a2 )
      goto LABEL_6;
  }
  else
  {
    if ( v5 != 21 )
      BUG();
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 192LL);
    v7 = *(_QWORD *)(v6 + 120);
    if ( v7 == a2 )
      goto LABEL_6;
  }
  if ( !(unsigned int)sub_8DED30(a2, v7, 5) )
  {
    sub_6854C0(147, a3, a1);
    return 1;
  }
LABEL_6:
  if ( (unsigned int)sub_8D32B0(a2) && (v8 = sub_8D46C0(a2), (unsigned int)sub_8D2310(v8))
    || (unsigned int)sub_8D3D10(a2) && (v10 = sub_8D4870(a2), (unsigned int)sub_8D2310(v10)) )
  {
    sub_6464A0(a2, a1, a3, 1u);
  }
  if ( !*(_QWORD *)(v6 + 256) )
    *(_QWORD *)(v6 + 256) = a2;
  *(_QWORD *)(v6 + 120) = sub_8D79B0(a2, *(_QWORD *)(v6 + 120));
  return 0;
}
