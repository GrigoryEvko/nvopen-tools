// Function: sub_87D630
// Address: 0x87d630
//
__int64 __fastcall sub_87D630(unsigned int a1, __int64 a2, __int64 a3)
{
  unsigned int i; // r12d
  __int64 v4; // r13
  __int64 v5; // rdi
  char v6; // al
  __int64 v7; // rax
  __int64 **v8; // rax
  unsigned int v9; // eax

  i = a1;
  if ( a2 )
  {
    v4 = *(_QWORD *)(a3 + 16);
    for ( i = sub_87D600(a1, *(unsigned __int8 *)(a3 + 25)); ; i = sub_87D600(i, *(unsigned __int8 *)(v7 + 25)) )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( *(_QWORD *)(a2 + 8) == v4 )
        break;
      while ( 1 )
      {
        v5 = *(_QWORD *)(v4 + 16);
        v6 = *(_BYTE *)(v5 + 96);
        if ( (v6 & 2) == 0 )
          break;
        if ( (v6 & 1) != 0 )
        {
          v7 = *(_QWORD *)(v5 + 112);
          if ( !*(_QWORD *)v7 )
            goto LABEL_10;
        }
        v8 = sub_72B780(v5);
        v9 = sub_87D630((unsigned __int8)i, v8[1], v8);
        v4 = *(_QWORD *)(v4 + 8);
        i = v9;
        if ( *(_QWORD *)(a2 + 8) == v4 )
          return i;
      }
      v7 = *(_QWORD *)(v5 + 112);
LABEL_10:
      ;
    }
  }
  return i;
}
