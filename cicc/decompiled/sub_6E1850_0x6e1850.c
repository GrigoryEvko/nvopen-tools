// Function: sub_6E1850
// Address: 0x6e1850
//
void __fastcall sub_6E1850(__int64 a1)
{
  __int64 v1; // rdx
  char v2; // r8
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rcx

  v1 = *(_QWORD *)(a1 + 88);
  if ( !v1 )
    goto LABEL_12;
  v2 = 0;
  v3 = unk_4D03C48;
  do
  {
    if ( !v3 )
      goto LABEL_8;
    if ( v1 != v3 )
    {
      v4 = v3;
      while ( 1 )
      {
        v5 = v4;
        v4 = *(_QWORD *)(v4 + 40);
        if ( !v4 )
          break;
        if ( v4 == v1 )
        {
          *(_QWORD *)(v5 + 40) = *(_QWORD *)(v1 + 40);
          goto LABEL_14;
        }
      }
LABEL_8:
      *(_QWORD *)(a1 + 88) = 0;
      goto LABEL_9;
    }
    v3 = *(_QWORD *)(v1 + 40);
    v2 = 1;
LABEL_14:
    *(_QWORD *)(v1 + 40) = 0;
LABEL_9:
    v1 = *(_QWORD *)(v1 + 48);
  }
  while ( v1 );
  if ( v2 )
    unk_4D03C48 = v3;
LABEL_12:
  *(_QWORD *)(a1 + 96) = 0;
}
