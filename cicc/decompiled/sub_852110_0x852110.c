// Function: sub_852110
// Address: 0x852110
//
size_t __fastcall sub_852110(_QWORD *a1)
{
  int v1; // eax
  char *v2; // rax
  FILE *v3; // rcx
  size_t result; // rax

  if ( fread(a1 + 1, 4u, 1u, qword_4F5FB48) != 1 )
    goto LABEL_11;
  v1 = *((_DWORD *)a1 + 2);
  if ( !v1 )
    return 0;
  if ( v1 == 1 )
  {
    if ( fread((char *)a1 + 12, 4u, 1u, qword_4F5FB48) == 1 && fread(a1 + 2, 1u, 1u, qword_4F5FB48) == 1 )
      goto LABEL_6;
LABEL_11:
    sub_851ED0();
  }
  if ( v1 != 2 )
    sub_721090();
  if ( fread((char *)a1 + 12, 4u, 1u, qword_4F5FB48) != 1 )
    goto LABEL_11;
LABEL_6:
  v2 = sub_852050();
  v3 = qword_4F5FB48;
  a1[3] = v2;
  result = fread(a1 + 4, 8u, 1u, v3);
  if ( result != 1 )
    goto LABEL_11;
  return result;
}
