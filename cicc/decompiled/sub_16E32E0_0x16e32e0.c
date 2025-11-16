// Function: sub_16E32E0
// Address: 0x16e32e0
//
const char *__fastcall sub_16E32E0(const char **a1, unsigned int *a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax
  unsigned __int8 v6; // al
  const char *v7; // rbx

  if ( *((_BYTE *)a1 + 17) != 1 || (v6 = *((_BYTE *)a1 + 16), v6 <= 1u) )
  {
LABEL_2:
    sub_16E2F40((__int64)a1, (__int64)a2);
    v4 = a2[2];
    if ( (unsigned int)v4 >= a2[3] )
    {
      sub_16CD150((__int64)a2, a2 + 4, 0, 1, v2, v3);
      v4 = a2[2];
    }
    *(_BYTE *)(*(_QWORD *)a2 + v4) = 0;
    return *(const char **)a2;
  }
  if ( v6 != 3 )
  {
    if ( v6 == 4 )
      return *(const char **)*a1;
    goto LABEL_2;
  }
  v7 = *a1;
  if ( *a1 )
    strlen(*a1);
  return v7;
}
