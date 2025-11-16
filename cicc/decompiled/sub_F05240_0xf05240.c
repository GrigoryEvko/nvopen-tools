// Function: sub_F05240
// Address: 0xf05240
//
__int64 __fastcall sub_F05240(void *a1)
{
  char **v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // rdi
  const void *v5; // r13
  size_t v6; // rdx
  size_t v7; // r12
  char *v8; // rax

  v1 = (char **)&off_497B3E0;
  v2 = sub_F05A00(a1);
  v4 = 7;
  v5 = (const void *)sub_F05340(v2, v3);
  v7 = v6;
  v8 = "invalid";
  while ( 1 )
  {
    if ( v4 >= v7 && (!v7 || !memcmp(&v8[v4 - v7], v5, v7)) )
      return *((unsigned int *)v1 + 16);
    v1 += 9;
    if ( v1 == (char **)&off_497BFB0 )
      break;
    v8 = *v1;
    v4 = (unsigned __int64)v1[1];
  }
  return 0;
}
