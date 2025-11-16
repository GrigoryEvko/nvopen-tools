// Function: sub_1F2EE00
// Address: 0x1f2ee00
//
void __fastcall sub_1F2EE00(char *src, char *a2)
{
  char *i; // rbx
  __int64 v3; // r12
  char *v4; // rcx
  float v5; // xmm0_4
  __int64 v6; // rdx
  char *v7; // rax

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; *(_QWORD *)src = v3 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)i;
        v4 = i;
        v5 = *(float *)(*(_QWORD *)i + 116LL);
        if ( v5 > *(float *)(*(_QWORD *)src + 116LL) )
          break;
        v6 = *((_QWORD *)i - 1);
        v7 = i - 8;
        if ( v5 > *(float *)(v6 + 116) )
        {
          do
          {
            *((_QWORD *)v7 + 1) = v6;
            v4 = v7;
            v6 = *((_QWORD *)v7 - 1);
            v7 -= 8;
          }
          while ( *(float *)(v3 + 116) > *(float *)(v6 + 116) );
        }
        i += 8;
        *(_QWORD *)v4 = v3;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(src + 8, src, i - src);
      i += 8;
    }
  }
}
