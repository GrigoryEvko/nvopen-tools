// Function: sub_2242C20
// Address: 0x2242c20
//
__int64 __fastcall sub_2242C20(__int64 a1, wchar_t *a2, __int64 a3, wchar_t *a4, __int64 a5)
{
  const wchar_t *v7; // r14
  const wchar_t *v8; // rbx
  const wchar_t *v9; // rbp
  wchar_t *v10; // r12
  unsigned int v11; // r15d
  const wchar_t *v12; // r14
  const wchar_t *v13; // rbx
  int v15; // eax
  int v16; // edx
  wchar_t *v17; // [rsp+0h] [rbp-58h]
  wchar_t *s; // [rsp+18h] [rbp-40h]

  if ( a2 == (wchar_t *)a3 )
  {
    v7 = (const wchar_t *)&unk_4FD67F8;
  }
  else
  {
    if ( !a2 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v7 = (const wchar_t *)sub_2242590(a2, a3);
  }
  s = (wchar_t *)v7;
  if ( a4 == (wchar_t *)a5 )
  {
    v17 = (wchar_t *)&unk_4FD67F8;
  }
  else
  {
    if ( !a4 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v17 = (wchar_t *)sub_2242590(a4, a5);
  }
  v8 = v17;
  v9 = &v7[*((_QWORD *)v7 - 3)];
  v10 = &v17[*((_QWORD *)v17 - 3)];
  while ( 1 )
  {
    v11 = sub_2255070(a1, v7, v8);
    if ( v11 )
      break;
    v12 = &v7[wcslen(v7)];
    v13 = &v8[wcslen(v8)];
    if ( v9 == v12 )
    {
      if ( v10 == v13 )
        break;
      if ( v9 == v12 )
      {
        v11 = -1;
        break;
      }
    }
    if ( v10 == v13 )
    {
      v11 = 1;
      break;
    }
    v7 = v12 + 1;
    v8 = v13 + 1;
  }
  if ( v17 - 6 != (wchar_t *)&unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v15 = _InterlockedExchangeAdd(v17 - 2, 0xFFFFFFFF);
    }
    else
    {
      v15 = *(v17 - 2);
      *(v17 - 2) = v15 - 1;
    }
    if ( v15 <= 0 )
      j_j___libc_free_0_2((unsigned __int64)(v17 - 6));
  }
  if ( s - 6 != (wchar_t *)&unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v16 = _InterlockedExchangeAdd(s - 2, 0xFFFFFFFF);
    }
    else
    {
      v16 = *(s - 2);
      *(s - 2) = v16 - 1;
    }
    if ( v16 <= 0 )
      j_j___libc_free_0_2((unsigned __int64)(s - 6));
  }
  return v11;
}
