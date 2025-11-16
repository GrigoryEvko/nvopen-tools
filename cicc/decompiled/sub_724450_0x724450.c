// Function: sub_724450
// Address: 0x724450
//
int sub_724450()
{
  char *v0; // r13
  size_t v1; // r14
  FILE *v2; // r12

  v0 = sub_723F40(0);
  v1 = strlen(v0);
  v2 = (FILE *)sub_685E40((char *)qword_4D04570, 1, 0, 0, 3562);
  fwrite(v0, 1u, v1, v2);
  return fclose(v2);
}
