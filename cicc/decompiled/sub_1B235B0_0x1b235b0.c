// Function: sub_1B235B0
// Address: 0x1b235b0
//
void __fastcall sub_1B235B0(_QWORD *src, _QWORD *a2)
{
  _QWORD *i; // rbx
  _QWORD *v4; // rdi
  __int64 v5; // r13
  __int64 v6; // r8
  __int64 v7; // r15
  __int64 v8; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    for ( i = src + 3; a2 != i; src[2] = v7 )
    {
      while ( 1 )
      {
        v5 = *i;
        if ( (int)sub_16AEA10(*i + 24LL, src[1] + 24LL) < 0 )
          break;
        v4 = i;
        i += 3;
        sub_1B233A0(v4);
        if ( a2 == i )
          return;
      }
      v6 = i[1];
      v7 = i[2];
      if ( src != i )
      {
        v8 = i[1];
        memmove(src + 3, src, (char *)i - (char *)src);
        v6 = v8;
      }
      i += 3;
      *src = v5;
      src[1] = v6;
    }
  }
}
