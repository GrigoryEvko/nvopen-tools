// Function: sub_CA12A0
// Address: 0xca12a0
//
const char *__fastcall sub_CA12A0(const char **a1, _QWORD *a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rax
  unsigned __int8 v6; // al
  const char *v7; // rbx

  if ( *((_BYTE *)a1 + 33) != 1 )
    goto LABEL_2;
  v6 = *((_BYTE *)a1 + 32);
  if ( v6 <= 1u )
    goto LABEL_2;
  switch ( v6 )
  {
    case 4u:
      return *(const char **)*a1;
    case 6u:
      return *a1;
    case 3u:
      v7 = *a1;
      if ( *a1 )
        strlen(*a1);
      return v7;
    default:
LABEL_2:
      sub_CA0EC0((__int64)a1, (__int64)a2);
      v4 = a2[1];
      if ( (unsigned __int64)(v4 + 1) > a2[2] )
      {
        sub_C8D290((__int64)a2, a2 + 3, v4 + 1, 1u, v2, v3);
        v4 = a2[1];
      }
      *(_BYTE *)(*a2 + v4) = 0;
      return (const char *)*a2;
  }
}
