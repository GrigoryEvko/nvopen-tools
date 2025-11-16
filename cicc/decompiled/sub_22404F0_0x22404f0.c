// Function: sub_22404F0
// Address: 0x22404f0
//
__int64 __fastcall sub_22404F0(_QWORD *a1, const wchar_t *a2, __int64 a3)
{
  __int64 v6; // r12
  wchar_t *v7; // rax
  wchar_t *v8; // rdi
  __int64 v9; // rax
  size_t v10; // rbx
  __int64 v11; // rbx
  __int64 (*v13)(); // rax

  if ( a3 > 0 )
  {
    v6 = 0;
    while ( 1 )
    {
      v7 = (wchar_t *)a1[6];
      v8 = (wchar_t *)a1[5];
      if ( v7 != v8 )
      {
        v9 = v7 - v8;
        v10 = a3 - v6;
        if ( v9 <= a3 - v6 )
          v10 = v9;
        if ( v10 )
        {
          wmemcpy(v8, a2, v10);
          v8 = (wchar_t *)a1[5];
        }
        v6 += v10;
        v11 = v10;
        a1[5] = &v8[v11];
        if ( a3 <= v6 )
          return v6;
        a2 = (const wchar_t *)((char *)a2 + v11 * 4);
      }
      v13 = *(__int64 (**)())(*a1 + 104LL);
      if ( v13 != sub_2240440 && ((unsigned int (__fastcall *)(_QWORD *, _QWORD))v13)(a1, *(unsigned int *)a2) != -1 )
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
