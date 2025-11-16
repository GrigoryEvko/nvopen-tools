// Function: sub_5F1DC0
// Address: 0x5f1dc0
//
void __fastcall sub_5F1DC0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rax

  v3 = *(_QWORD *)(a2 + 288);
  if ( *(_BYTE *)(v3 + 140) == 12 && *(_QWORD *)(v3 + 8) )
  {
    do
      v3 = *(_QWORD *)(v3 + 160);
    while ( *(_BYTE *)(v3 + 140) == 12 );
    v4 = *(_QWORD *)(v3 + 168);
    if ( a3 && (*(_WORD *)(v4 + 18) & 0x3FFF) != 0 )
    {
      v5 = sub_7259C0(7);
      *(_QWORD *)(a2 + 288) = v5;
      sub_73C230(v3, v5);
      v6 = *(_QWORD *)(*(_QWORD *)(a2 + 288) + 168LL);
      goto LABEL_8;
    }
    if ( !*(_QWORD *)(v4 + 56) )
    {
      if ( (v7 = *(_BYTE *)(a1 + 16), (v7 & 8) != 0) && ((*(_BYTE *)(a1 + 56) - 2) & 0xFD) == 0 || (v7 & 0x20) != 0 )
      {
        v8 = sub_7259C0(7);
        *(_QWORD *)(a2 + 288) = v8;
        sub_73C230(v3, v8);
        if ( a3 )
        {
          v6 = *(_QWORD *)(*(_QWORD *)(a2 + 288) + 168LL);
LABEL_8:
          if ( (unsigned int)sub_8D3150() )
          {
            sub_6851C0(988, a1 + 8);
            *(_WORD *)(v6 + 18) = 0;
          }
        }
      }
    }
  }
}
