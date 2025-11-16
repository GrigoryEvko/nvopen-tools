// Function: sub_723280
// Address: 0x723280
//
int sub_723280()
{
  _QWORD *v0; // rax
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // r14
  char *v5; // rax
  char *v6; // rax
  char *v7; // rbx
  char *v8; // rax
  char *v9; // rbx
  char *v10; // rax
  __int64 v11; // r12
  char *v12; // rax
  FILE *v13; // r12
  char *v14; // rax
  char *v15; // rax
  char *v16; // rax
  char *v17; // rax
  char *v18; // rbx
  char *v19; // rax
  char *v20; // rbx
  char *v21; // rax

  LODWORD(v0) = dword_4D04198;
  if ( dword_4D04198 )
  {
    if ( dword_4D04198 == 1 )
      LODWORD(v0) = fwrite("]}]}\n", 1u, 5u, qword_4F07510);
    return (int)v0;
  }
  v0 = &qword_4F074A0;
  v1 = qword_4F074B0;
  v2 = qword_4F074B8;
  if ( !(qword_4F074B8 + qword_4F074B0) )
    return (int)v0;
  v3 = *(_QWORD *)&dword_4F074D0;
  v4 = *(_QWORD *)&dword_4F074D0 + *(_QWORD *)&dword_4F074D8;
  if ( (_DWORD)qword_4F074B8 + (_DWORD)qword_4F074B0 == dword_4F074D0 + dword_4F074D8 )
  {
    if ( v4 && qword_4F074B0 )
    {
      if ( *(_QWORD *)&dword_4F074D0 == 1 )
      {
        v18 = sub_67C860(1742);
        v19 = sub_67C860(3235);
        fprintf(qword_4F07510, "%lu %s %s", 1, v19, v18);
      }
      else
      {
        v9 = sub_67C860(1743);
        v10 = sub_67C860(3235);
        fprintf(qword_4F07510, "%lu %s %s", v3, v10, v9);
      }
    }
    goto LABEL_18;
  }
  if ( qword_4F074B0 )
  {
    v5 = sub_67C860((unsigned int)(qword_4F074B0 != 1) + 1742);
    fprintf(qword_4F07510, "%lu %s", v1, v5);
    if ( !v2 )
    {
      if ( !v4 )
        goto LABEL_18;
      v6 = sub_67C860(3234);
      fprintf(qword_4F07510, " (%s ", v6);
LABEL_11:
      if ( v3 == 1 )
      {
        v20 = sub_67C860(1742);
        v21 = sub_67C860(3235);
        fprintf(qword_4F07510, "%lu %s %s", 1, v21, v20);
      }
      else
      {
        v7 = sub_67C860(1743);
        v8 = sub_67C860(3235);
        fprintf(qword_4F07510, "%lu %s %s", v3, v8, v7);
      }
      goto LABEL_29;
    }
    v17 = sub_67C860(1746);
    fprintf(qword_4F07510, " %s ", v17);
  }
  v15 = sub_67C860((unsigned int)(v2 != 1) + 1744);
  fprintf(qword_4F07510, "%lu %s", v2, v15);
  if ( !v4 )
    goto LABEL_18;
  v16 = sub_67C860(3234);
  fprintf(qword_4F07510, " (%s ", v16);
  if ( v1 )
    goto LABEL_11;
LABEL_29:
  fputc(41, qword_4F07510);
LABEL_18:
  fputc(32, qword_4F07510);
  if ( qword_4F076F0 && *qword_4F076F0 && (*qword_4F076F0 != 45 || qword_4F076F0[1]) )
  {
    if ( qword_4D046E0 )
      v11 = sub_723260(qword_4D046E0);
    else
      v11 = sub_723260(qword_4F076F0);
    v12 = sub_67C860(1747);
    fprintf(qword_4F07510, v12, v11);
  }
  else
  {
    v13 = qword_4F07510;
    v14 = sub_67C860(1748);
    fputs(v14, v13);
  }
  LODWORD(v0) = fputc(10, qword_4F07510);
  return (int)v0;
}
