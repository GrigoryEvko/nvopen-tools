// Function: sub_5C6D70
// Address: 0x5c6d70
//
void __fastcall sub_5C6D70(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  __int64 i; // rdi
  __int64 v4; // rax
  __int64 j; // rax
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v6[0] = a2;
  if ( a1 )
  {
    v2 = a1;
    do
    {
      for ( i = v2[1]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( !(unsigned int)sub_8D23B0() && unk_4F04C44 == -1 )
      {
        v4 = unk_4F04C68 + 776LL * unk_4F04C64;
        if ( (*(_BYTE *)(v4 + 6) & 6) == 0 && *(_BYTE *)(v4 + 4) != 12 )
        {
          for ( j = v2[1]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( !*(_QWORD *)(j + 128) )
            sub_685360(3669, v6);
        }
      }
      v2 = (_QWORD *)*v2;
    }
    while ( v2 );
  }
}
