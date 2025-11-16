// Function: sub_6E5A30
// Address: 0x6e5a30
//
void __fastcall sub_6E5A30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax

  if ( a1 )
  {
    v4 = a1;
    do
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)v4;
        if ( a2 == 4 )
          break;
        if ( (a2 & v6) != 0 )
        {
          *(_BYTE *)(v4 + 8) = 0;
          v5 = a3 | v6 & 0xFFFFFFFFFFFECF87LL;
          *(_QWORD *)v4 = v5;
          if ( (v5 & 0x20) != 0 )
            goto LABEL_9;
        }
LABEL_5:
        v4 = *(_QWORD *)(v4 + 48);
        if ( !v4 )
          return;
      }
      if ( (v6 & 0x13078) != 0 )
        goto LABEL_5;
      v7 = a3 | v6;
      *(_BYTE *)(v4 + 8) = 0;
      *(_QWORD *)v4 = v7;
      if ( (v7 & 0x20) == 0 )
        goto LABEL_5;
LABEL_9:
      sub_6E5700((unsigned __int64 *)v4);
      v4 = *(_QWORD *)(v4 + 48);
    }
    while ( v4 );
  }
}
