// Function: sub_22407C0
// Address: 0x22407c0
//
__int64 __fastcall sub_22407C0(_QWORD *a1, wchar_t *a2, __int64 a3)
{
  __int64 v5; // r12
  const wchar_t *v6; // rax
  const wchar_t *v7; // rsi
  __int64 v8; // rax
  size_t v9; // rbx
  __int64 v10; // rbx
  wchar_t *v12; // rbx
  __int64 (__fastcall *v13)(_QWORD *); // rdx
  __int64 (*v14)(); // rax
  wchar_t *v15; // rdx
  wchar_t v16; // eax

  if ( a3 > 0 )
  {
    v5 = 0;
    while ( 1 )
    {
      v6 = (const wchar_t *)a1[3];
      v7 = (const wchar_t *)a1[2];
      if ( v6 == v7 )
      {
        v12 = a2;
      }
      else
      {
        v8 = v6 - v7;
        v9 = a3 - v5;
        if ( v8 <= a3 - v5 )
          v9 = v8;
        if ( v9 )
        {
          wmemcpy(a2, v7, v9);
          v7 = (const wchar_t *)a1[2];
        }
        v5 += v9;
        v10 = v9;
        a1[2] = &v7[v10];
        if ( a3 <= v5 )
          return v5;
        v12 = &a2[v10];
      }
      v13 = *(__int64 (__fastcall **)(_QWORD *))(*a1 + 80LL);
      if ( v13 == sub_22406A0 )
      {
        v14 = *(__int64 (**)())(*a1 + 72LL);
        if ( v14 == sub_2240420 || ((unsigned int (__fastcall *)(_QWORD *))v14)(a1) == -1 )
          return v5;
        v15 = (wchar_t *)a1[2];
        v16 = *v15;
        a1[2] = v15 + 1;
      }
      else
      {
        v16 = v13(a1);
      }
      if ( v16 != -1 )
      {
        ++v5;
        *v12 = v16;
        a2 = v12 + 1;
        if ( a3 > v5 )
          continue;
      }
      return v5;
    }
  }
  return 0;
}
