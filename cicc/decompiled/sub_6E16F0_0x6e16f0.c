// Function: sub_6E16F0
// Address: 0x6e16f0
//
void __fastcall sub_6E16F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  v2 = unk_4D03C48;
  unk_4D03C48 = 0;
  if ( v2 )
  {
    v3 = 0;
    do
    {
      v4 = v2;
      v2 = *(_QWORD *)(v2 + 40);
      if ( a1 )
      {
        if ( a1 == v4 )
          goto LABEL_13;
        v5 = a1;
        while ( 1 )
        {
          v5 = *(_QWORD *)(v5 + 48);
          if ( !v5 )
            break;
          if ( v5 == v4 )
            goto LABEL_13;
        }
      }
      if ( !a2 )
        goto LABEL_19;
      if ( a2 != v4 )
      {
        v6 = a2;
        while ( 1 )
        {
          v6 = *(_QWORD *)(v6 + 48);
          if ( !v6 )
            break;
          if ( v6 == v4 )
            goto LABEL_13;
        }
LABEL_19:
        if ( !*(_BYTE *)(v4 + 8) )
        {
          sub_875E10(*(_QWORD *)v4, *(_QWORD *)(v4 + 16), v4 + 32, 1, *(_QWORD *)(v4 + 24));
          *(_BYTE *)(v4 + 8) = 1;
        }
        v7 = qword_4D03A98;
        qword_4D03A98 = v4;
        *(_QWORD *)(v4 + 40) = v7;
        continue;
      }
LABEL_13:
      if ( unk_4D03C48 )
        *(_QWORD *)(v3 + 40) = v4;
      else
        unk_4D03C48 = v4;
      *(_QWORD *)(v4 + 40) = 0;
      v3 = v4;
    }
    while ( v2 );
  }
}
