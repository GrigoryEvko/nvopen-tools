// Function: sub_5C7880
// Address: 0x5c7880
//
__int64 __fastcall sub_5C7880(char *s, char a2)
{
  char *v2; // r12
  size_t v4; // rax
  size_t v5; // rbx
  char dest[128]; // [rsp+0h] [rbp-80h] BYREF

  v2 = s;
  if ( *s == 95
    && s[1] == 95
    && (unk_4F077C4 != 2 && unk_4F07778 > 202310 || a2 == 2 || a2 == 5 || a2 == 1 && unk_4F077B8) )
  {
    v4 = strlen(s);
    if ( v4 > 4 && s[v4 - 1] == 95 && s[v4 - 2] == 95 )
    {
      v5 = v4 - 4;
      if ( v4 - 4 <= 0x64 )
      {
        v2 = dest;
        strncpy(dest, s + 2, v4 - 4);
        dest[v5] = 0;
      }
    }
  }
  return sub_881B20(qword_4CF79B0, v2, 0);
}
