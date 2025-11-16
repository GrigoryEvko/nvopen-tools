// Function: sub_22406F0
// Address: 0x22406f0
//
__int64 __fastcall sub_22406F0(_QWORD *a1, void *a2, __int64 a3)
{
  __int64 v5; // r12
  const void *v6; // rsi
  __int64 v7; // rax
  size_t v8; // rbx
  char *v9; // rax
  char *v11; // rbx
  __int64 (__fastcall *v12)(_QWORD *); // rdx
  __int64 (*v13)(); // rax
  char *v14; // rax
  char v15; // dl
  int v16; // eax

  if ( a3 > 0 )
  {
    v5 = 0;
    while ( 1 )
    {
      v6 = (const void *)a1[2];
      v7 = a1[3] - (_QWORD)v6;
      if ( v7 )
      {
        v8 = a3 - v5;
        if ( a3 - v5 > v7 )
          v8 = a1[3] - (_QWORD)v6;
        v5 += v8;
        v9 = (char *)memcpy(a2, v6, v8);
        a1[2] += v8;
        if ( a3 <= v5 )
          return v5;
        v11 = &v9[v8];
      }
      else
      {
        v11 = (char *)a2;
      }
      v12 = *(__int64 (__fastcall **)(_QWORD *))(*a1 + 80LL);
      if ( v12 == sub_2240650 )
      {
        v13 = *(__int64 (**)())(*a1 + 72LL);
        if ( v13 == sub_2240390 || ((unsigned int (__fastcall *)(_QWORD *))v13)(a1) == -1 )
          return v5;
        v14 = (char *)a1[2];
        v15 = *v14;
        a1[2] = v14 + 1;
      }
      else
      {
        v16 = v12(a1);
        v15 = v16;
        if ( v16 == -1 )
          return v5;
      }
      ++v5;
      *v11 = v15;
      a2 = v11 + 1;
      if ( a3 <= v5 )
        return v5;
    }
  }
  return 0;
}
