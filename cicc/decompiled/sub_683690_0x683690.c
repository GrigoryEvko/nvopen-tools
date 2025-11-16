// Function: sub_683690
// Address: 0x683690
//
__int64 __fastcall sub_683690(__int64 a1)
{
  __int64 v1; // rdi
  __int64 v2; // rax
  char *v3; // rbx
  char v4; // dl
  _QWORD *v5; // rdi
  __int64 v6; // rax
  char v7; // cl

  sub_8238B0(qword_4D039D8, "{\"text\":\"", 9);
  sub_681B50(a1);
  v1 = qword_4D039E8;
  v2 = *(_QWORD *)(qword_4D039E8 + 16);
  if ( (unsigned __int64)(v2 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
  {
    sub_823810(qword_4D039E8);
    v1 = qword_4D039E8;
    v2 = *(_QWORD *)(qword_4D039E8 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v1 + 32) + v2) = 0;
  v3 = *(char **)(v1 + 32);
  ++*(_QWORD *)(v1 + 16);
  v4 = *v3;
  if ( *v3 )
  {
    v5 = (_QWORD *)qword_4D039D8;
    v6 = *(_QWORD *)(qword_4D039D8 + 16);
    while ( v4 != 34 && v4 != 92 )
    {
      if ( (unsigned __int64)(v6 + 1) > v5[1] )
        goto LABEL_12;
LABEL_6:
      v7 = *v3++;
      *(_BYTE *)(v5[4] + v6) = v7;
      v6 = v5[2] + 1LL;
      v5[2] = v6;
      v4 = *v3;
      if ( !*v3 )
      {
        v1 = qword_4D039E8;
        goto LABEL_14;
      }
    }
    if ( (unsigned __int64)(v6 + 1) > v5[1] )
    {
      sub_823810(v5);
      v5 = (_QWORD *)qword_4D039D8;
      v6 = *(_QWORD *)(qword_4D039D8 + 16);
    }
    *(_BYTE *)(v5[4] + v6) = 92;
    v6 = v5[2] + 1LL;
    v5[2] = v6;
    if ( (unsigned __int64)(v6 + 1) <= v5[1] )
      goto LABEL_6;
LABEL_12:
    sub_823810(v5);
    v5 = (_QWORD *)qword_4D039D8;
    v6 = *(_QWORD *)(qword_4D039D8 + 16);
    goto LABEL_6;
  }
LABEL_14:
  sub_823800(v1);
  return sub_8238B0(qword_4D039D8, "\"}", 2);
}
