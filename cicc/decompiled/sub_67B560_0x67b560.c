// Function: sub_67B560
// Address: 0x67b560
//
size_t __fastcall sub_67B560(unsigned __int8 a1, const char *a2)
{
  char *v2; // rbx
  size_t result; // rax
  const char *v4; // rbx
  char *v5; // rax
  size_t v6; // rdx
  const char *v7; // rdx

  v2 = strstr(unk_4F073C0, a2);
  result = 16LL * a1 + 82867168;
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  if ( v2 )
  {
    result = strlen(a2);
    if ( v2[result] == 61 )
    {
      v4 = &v2[result + 1];
      v5 = strchr(v4, 58);
      v6 = v5 - v4;
      if ( !v5 )
        v6 = strlen(v4);
      result = 16LL * a1 + 82867168;
      *(_QWORD *)(result + 8) = v6;
      v7 = &v4[v6];
      *(_QWORD *)result = v4;
      if ( v4 < v7 )
      {
        while ( 1 )
        {
          result = *(unsigned __int8 *)v4;
          if ( (_BYTE)result != 59 )
          {
            result = (unsigned int)(result - 48);
            if ( (unsigned int)result > 9 )
              break;
          }
          if ( ++v4 == v7 )
            return result;
        }
        qword_4F073E0[2 * a1] = 0;
        return (size_t)qword_4F073E0;
      }
    }
  }
  return result;
}
