// Function: sub_22405B0
// Address: 0x22405b0
//
__int64 __fastcall sub_22405B0(_QWORD *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v6; // r12
  void *v7; // rdi
  __int64 v8; // rax
  size_t v9; // rbp
  __int64 (*v11)(); // rax

  if ( a3 > 0 )
  {
    v6 = 0;
    while ( 1 )
    {
      v7 = (void *)a1[5];
      v8 = a1[6] - (_QWORD)v7;
      if ( v8 )
      {
        v9 = a3 - v6;
        if ( a3 - v6 > v8 )
          v9 = a1[6] - (_QWORD)v7;
        v6 += v9;
        memcpy(v7, a2, v9);
        a1[5] += v9;
        if ( a3 <= v6 )
          return v6;
        a2 += v9;
      }
      v11 = *(__int64 (**)())(*a1 + 104LL);
      if ( v11 != sub_22403B0 && ((unsigned int (__fastcall *)(_QWORD *, _QWORD))v11)(a1, *a2) != -1 )
      {
        ++v6;
        ++a2;
        if ( a3 > v6 )
          continue;
      }
      return v6;
    }
  }
  return 0;
}
