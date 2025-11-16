// Function: sub_7FA8C0
// Address: 0x7fa8c0
//
__int64 __fastcall sub_7FA8C0(__int64 a1)
{
  __int64 v1; // r13
  char v3; // al
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v7; // rax
  __int64 v8; // r12

  v1 = a1;
  v3 = *(_BYTE *)(a1 + 48);
  if ( v3 == 6 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 176LL);
    if ( !v7 || *(_BYTE *)(v7 + 173) != 11 )
      return 0;
    v1 = *(_QWORD *)(*(_QWORD *)(v7 + 176) + 176LL);
    v3 = *(_BYTE *)(v1 + 48);
  }
  if ( v3 != 5 )
  {
LABEL_3:
    if ( (v3 & 0xFD) != 0 )
      return 0;
    goto LABEL_4;
  }
  v8 = *(_QWORD *)(v1 + 56);
  if ( sub_7E5340(v8) || !sub_7F7E80(v8) )
  {
    v3 = *(_BYTE *)(v1 + 48);
    goto LABEL_3;
  }
LABEL_4:
  v4 = *(_QWORD *)(a1 + 16);
  if ( !v4 )
  {
    if ( *(_BYTE *)(v1 + 48) != 5 )
      return 1;
LABEL_15:
    sub_7F6D70(*(_QWORD *)(v1 + 56), *(_BYTE **)(a1 + 8));
LABEL_8:
    v5 = *(_QWORD *)(a1 + 16);
    if ( v5 )
      sub_7F6D70(v5, *(_BYTE **)(a1 + 8));
    return 1;
  }
  if ( !sub_7E5340(*(_QWORD *)(a1 + 16)) && sub_7F7E80(v4) )
  {
    if ( *(_BYTE *)(v1 + 48) != 5 )
      goto LABEL_8;
    goto LABEL_15;
  }
  return 0;
}
