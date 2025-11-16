// Function: sub_862730
// Address: 0x862730
//
char *__fastcall sub_862730(int a1, int a2)
{
  char *result; // rax
  char *v3; // rdi
  __int64 v4; // r12

  result = (char *)&qword_4F07280;
  v3 = (char *)qword_4F072B8 + 16 * a1;
  v4 = *(_QWORD *)v3;
  if ( *(_QWORD *)v3 )
  {
    result = (char *)qword_4F073B0;
    if ( *((_QWORD *)qword_4F073B0 + *((int *)v3 + 2)) )
    {
      if ( *(_BYTE *)(v4 + 28) == 17 )
      {
        if ( !a2 || (result = *(char **)(v4 + 32), result[192] < 0) )
        {
          if ( (*(_BYTE *)(v4 - 8) & 2) == 0 && (*(_BYTE *)(v4 + 29) & 1) == 0 )
          {
            if ( dword_4F077C4 == 2 && (*(_BYTE *)(*(_QWORD *)(v4 + 32) + 202LL) & 0x20) != 0 )
              sub_8627E0(*(_QWORD *)v3);
            return (char *)sub_85B450(v4, 0, 1);
          }
        }
      }
    }
  }
  return result;
}
