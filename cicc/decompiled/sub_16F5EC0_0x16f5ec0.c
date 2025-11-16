// Function: sub_16F5EC0
// Address: 0x16f5ec0
//
__int64 __fastcall sub_16F5EC0(__int64 a1, unsigned __int64 a2)
{
  _UNKNOWN **v2; // rbx
  char *v3; // rax
  __int64 v4; // rdx
  char *v5; // r13
  size_t v6; // rdx
  size_t v7; // r12
  char *v8; // rdi
  unsigned int v9; // r15d

  v2 = &off_4CD4080;
  v3 = (char *)sub_16F5C40(a1, a2);
  v5 = sub_16F5560(v3, v4);
  v7 = v6;
  while ( 1 )
  {
    v8 = (char *)v2[1];
    if ( (unsigned __int64)v8 >= v7 )
    {
      v9 = *((_DWORD *)v2 + 14);
      if ( !v7 || !memcmp(&v8[(_QWORD)*v2 - v7], v5, v7) )
        break;
    }
    v2 += 8;
    if ( v2 == (_UNKNOWN **)&unk_4CD48C0 )
      return 0;
  }
  return v9;
}
