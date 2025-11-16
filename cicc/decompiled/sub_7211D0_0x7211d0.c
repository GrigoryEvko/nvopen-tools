// Function: sub_7211D0
// Address: 0x7211d0
//
int __fastcall sub_7211D0(const char *a1, unsigned __int64 *a2, __int64 *a3)
{
  double v4; // xmm1_8
  double v5; // xmm0_8
  unsigned __int64 v6; // rax
  double v7; // xmm2_8
  __int64 v9; // rdx

  v4 = difftime(a3[1], a2[1]);
  if ( *a3 < 0 )
  {
    v9 = *a3 & 1 | ((unsigned __int64)*a3 >> 1);
    v6 = *a2;
    v5 = (double)(int)v9 + (double)(int)v9;
    if ( (*a2 & 0x8000000000000000LL) == 0LL )
      goto LABEL_3;
LABEL_6:
    v7 = (double)(int)(v6 & 1 | (v6 >> 1)) + (double)(int)(v6 & 1 | (v6 >> 1));
    return fprintf(qword_4F07510, "%-30s %10.2f (CPU) %10.2f (elapsed)\n", a1, (v5 - v7) / 1000.0, v4);
  }
  v5 = (double)(int)*a3;
  v6 = *a2;
  if ( (*a2 & 0x8000000000000000LL) != 0LL )
    goto LABEL_6;
LABEL_3:
  v7 = (double)(int)v6;
  return fprintf(qword_4F07510, "%-30s %10.2f (CPU) %10.2f (elapsed)\n", a1, (v5 - v7) / 1000.0, v4);
}
