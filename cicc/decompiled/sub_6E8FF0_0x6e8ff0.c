// Function: sub_6E8FF0
// Address: 0x6e8ff0
//
__int64 __fastcall sub_6E8FF0(__int64 a1, _BYTE *a2, _DWORD *a3, _QWORD *a4)
{
  __int64 v7; // rdi
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // rax
  char i; // dl
  __int64 v12; // r15
  int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // rcx
  int v17; // eax

  v7 = *(_QWORD *)a1;
  v8 = *(_BYTE *)(v7 + 140);
  if ( v8 == 12 )
  {
    v9 = v7;
    do
    {
      v9 = *(_QWORD *)(v9 + 160);
      v8 = *(_BYTE *)(v9 + 140);
    }
    while ( v8 == 12 );
  }
  if ( !v8 )
    goto LABEL_8;
  v10 = *(_QWORD *)a2;
  for ( i = *(_BYTE *)(*(_QWORD *)a2 + 140LL); i == 12; i = *(_BYTE *)(v10 + 140) )
    v10 = *(_QWORD *)(v10 + 160);
  if ( !i )
  {
LABEL_8:
    LODWORD(v12) = 1;
    *a4 = sub_72C930(v7);
    return (unsigned int)v12;
  }
  v14 = sub_8D2690(v7);
  v15 = *(_QWORD *)a2;
  if ( v14 || (v17 = sub_8D2690(v15), v15 = *(_QWORD *)a1, v17) )
  {
    LODWORD(v12) = sub_8D2660(v15);
  }
  else if ( (unsigned int)sub_8D2660(v15) )
  {
    if ( (unsigned int)sub_8D2660(*(_QWORD *)a2) )
    {
      v16 = *(_QWORD *)a2;
      LODWORD(v12) = 1;
      goto LABEL_13;
    }
    if ( a2[16] != 2 )
      goto LABEL_18;
    v12 = (unsigned int)sub_712690(a2 + 144) != 0;
  }
  else
  {
    if ( *(_BYTE *)(a1 + 16) != 2 )
    {
LABEL_18:
      v16 = *(_QWORD *)a2;
      goto LABEL_19;
    }
    v12 = (unsigned int)sub_712690(a1 + 144) != 0;
  }
  v16 = *(_QWORD *)a2;
  if ( !(_DWORD)v12 )
  {
LABEL_19:
    LODWORD(v12) = 0;
    sub_6E5ED0(0x2Au, a3, *(_QWORD *)a1, v16);
    *a4 = sub_72C930(42);
    return (unsigned int)v12;
  }
LABEL_13:
  if ( (unsigned int)sub_8D2690(v16) )
    *a4 = sub_72C4C0();
  else
    *a4 = sub_72C570();
  return (unsigned int)v12;
}
