// Function: sub_16485A0
// Address: 0x16485a0
//
_QWORD *__fastcall sub_16485A0(_QWORD *a1, _QWORD *a2)
{
  unsigned int *v2; // rax
  __int64 v4; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  char v8; // cl

  v2 = (unsigned int *)&unk_42ABE60;
  while ( 1 )
  {
    a2 -= 3;
    if ( a1 == a2 + 3 )
      break;
    v4 = *v2++;
    *a2 = 0;
    a2[2] = v4;
    if ( v2 == (unsigned int *)jpt_1649193 )
    {
      if ( a2 != a1 )
      {
        v6 = 20;
        v7 = 20;
        do
        {
          a2 -= 3;
          ++v7;
          *a2 = 0;
          if ( v6 )
          {
            v8 = v6;
            v6 >>= 1;
            a2[2] = v8 & 1;
          }
          else
          {
            a2[2] = 2;
            v6 = v7;
          }
        }
        while ( a1 != a2 );
      }
      return a1;
    }
  }
  return a1;
}
