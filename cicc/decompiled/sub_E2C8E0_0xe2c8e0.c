// Function: sub_E2C8E0
// Address: 0xe2c8e0
//
unsigned __int64 __fastcall sub_E2C8E0(__int64 a1, char **a2, unsigned int a3, size_t a4, const void *a5)
{
  unsigned __int64 result; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // rbx
  __int64 v12; // rdi
  char *v13; // rax
  size_t v14; // rdx
  char *v15; // rdi
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rdx
  __int64 v18; // rax

  result = *(_QWORD *)(a1 + 24);
  if ( result )
  {
    v10 = **(_QWORD **)(a1 + 16);
    if ( v10 )
    {
      (*(void (__fastcall **)(__int64, char **, _QWORD))(*(_QWORD *)v10 + 16LL))(v10, a2, a3);
      result = *(_QWORD *)(a1 + 24);
    }
    if ( result > 1 )
    {
      v11 = 1;
      do
      {
        if ( a4 )
        {
          v13 = a2[1];
          v14 = (size_t)a2[2];
          v15 = *a2;
          if ( (unsigned __int64)&v13[a4] > v14 )
          {
            v16 = (unsigned __int64)&v13[a4 + 992];
            v17 = 2 * v14;
            if ( v16 > v17 )
              a2[2] = (char *)v16;
            else
              a2[2] = (char *)v17;
            v18 = realloc(v15);
            *a2 = (char *)v18;
            v15 = (char *)v18;
            if ( !v18 )
              abort();
            v13 = a2[1];
          }
          memcpy(&v15[(_QWORD)v13], a5, a4);
          a2[1] += a4;
        }
        v12 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v11++);
        result = (*(__int64 (__fastcall **)(__int64, char **, _QWORD))(*(_QWORD *)v12 + 16LL))(v12, a2, a3);
      }
      while ( *(_QWORD *)(a1 + 24) > v11 );
    }
  }
  return result;
}
