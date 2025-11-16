// Function: sub_BD83B0
// Address: 0xbd83b0
//
void __fastcall sub_BD83B0(__int64 a1, __int64 a2, int a3)
{
  __int64 i; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 *v8; // rdi

  if ( (*(_BYTE *)(a1 + 1) & 1) != 0 )
    sub_BD7FF0(a1, a2);
  if ( a3 == 1 && (*(_BYTE *)(a1 + 7) & 8) != 0 )
    sub_BA6240(a1, (_BYTE *)a2);
  for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(a1 + 16) )
  {
    while ( 1 )
    {
      v8 = *(__int64 **)(i + 24);
      if ( (unsigned __int8)(*(_BYTE *)v8 - 4) > 0x11u )
        break;
      sub_ADBE50(v8, (__int64 *)a1, (_BYTE *)a2);
      i = *(_QWORD *)(a1 + 16);
      if ( !i )
        goto LABEL_16;
    }
    if ( *(_QWORD *)i )
    {
      v6 = *(_QWORD *)(i + 8);
      **(_QWORD **)(i + 16) = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(i + 16);
    }
    *(_QWORD *)i = a2;
    if ( a2 )
    {
      v7 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(i + 8) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = i + 8;
      *(_QWORD *)(i + 16) = a2 + 16;
      *(_QWORD *)(a2 + 16) = i;
    }
  }
LABEL_16:
  if ( *(_BYTE *)a1 == 23 )
    sub_AA5E80(a1, a2);
}
