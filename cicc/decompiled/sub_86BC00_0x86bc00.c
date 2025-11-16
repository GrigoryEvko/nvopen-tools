// Function: sub_86BC00
// Address: 0x86bc00
//
_BOOL8 __fastcall sub_86BC00(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  char v3; // al
  char v4; // al
  __int64 v5; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8

  i = a1;
  v3 = *(_BYTE *)(a1 + 24);
  if ( v3 == 1 )
  {
    v4 = *(_BYTE *)(a1 + 56);
    if ( v4 == 20 )
      goto LABEL_11;
LABEL_3:
    if ( v4 == 91
      || (unsigned __int8)(v4 - 105) <= 4u
      && (a2 = 0, (v5 = sub_72B0F0(*(_QWORD *)(i + 72), 0)) != 0)
      && !*(_BYTE *)(v5 + 174)
      && (unsigned __int16)(*(_WORD *)(v5 + 176) - 4686) <= 1u )
    {
      for ( i = *(_QWORD *)(*(_QWORD *)(i + 72) + 16LL); ; i = *(_QWORD *)(i + 72) )
      {
        v3 = *(_BYTE *)(i + 24);
        if ( v3 != 1 )
          break;
        v4 = *(_BYTE *)(i + 56);
        if ( v4 != 20 )
          goto LABEL_3;
LABEL_11:
        ;
      }
    }
    else
    {
      v3 = *(_BYTE *)(i + 24);
    }
  }
  return v3 == 2
      && sub_70FCE0(*(_QWORD *)(i + 56))
      && (unsigned int)sub_711520(*(_QWORD *)(i + 56), a2, v7, v8, v9) == 0;
}
