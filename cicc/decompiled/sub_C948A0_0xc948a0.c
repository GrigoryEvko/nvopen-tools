// Function: sub_C948A0
// Address: 0xc948a0
//
char *__fastcall sub_C948A0(char ***a1, const void *a2, size_t a3)
{
  char **v3; // rdi
  char *v5; // r8
  char *v6; // rax

  v3 = *a1;
  v5 = *v3;
  v3[10] += a3 + 1;
  v6 = &v5[a3 + 1];
  if ( v3[1] >= v6 && v5 )
    *v3 = v6;
  else
    v5 = (char *)sub_9D1E70((__int64)v3, a3 + 1, a3 + 1, 0);
  if ( a3 )
    v5 = (char *)memcpy(v5, a2, a3);
  v5[a3] = 0;
  return v5;
}
