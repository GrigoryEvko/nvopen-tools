// Function: sub_12EC8D0
// Address: 0x12ec8d0
//
__int64 __fastcall sub_12EC8D0(__int64 a1, const void *a2, size_t a3)
{
  const char *v4; // r12
  unsigned int *v5; // rbx

  v4 = "help";
  v5 = (unsigned int *)&unk_497C3C0;
  if ( "+help" == (char *)-1LL )
    goto LABEL_5;
LABEL_2:
  if ( strlen(v4) == a3 && (!a3 || !memcmp(a2, v4, a3)) )
    return v5[8];
  while ( 1 )
  {
    v5 += 16;
    if ( v5 == (unsigned int *)&unk_497FB00 )
      return 0;
    v4 = (const char *)*((_QWORD *)v5 + 1);
    if ( v4 )
      goto LABEL_2;
LABEL_5:
    if ( !a3 )
      return v5[8];
  }
}
