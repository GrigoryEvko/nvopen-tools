// Function: sub_222F1E0
// Address: 0x222f1e0
//
__int64 __fastcall sub_222F1E0(__int64 a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  const char *v7; // r14
  const char *v8; // rbx
  const char *v9; // rbp
  char *v10; // r12
  unsigned int v11; // r15d
  const char *v12; // r14
  const char *v13; // rbx
  int v15; // eax
  int v16; // edx
  char *v17; // [rsp+0h] [rbp-58h]
  char *s; // [rsp+18h] [rbp-40h]

  if ( a2 == (_BYTE *)a3 )
  {
    v7 = (const char *)&unk_4FD67D8;
  }
  else
  {
    if ( !a2 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v7 = (const char *)sub_222EC60(a2, a3);
  }
  s = (char *)v7;
  if ( a4 == (_BYTE *)a5 )
  {
    v17 = (char *)&unk_4FD67D8;
  }
  else
  {
    if ( !a4 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v17 = (char *)sub_222EC60(a4, a5);
  }
  v8 = v17;
  v9 = &v7[*((_QWORD *)v7 - 3)];
  v10 = &v17[*((_QWORD *)v17 - 3)];
  while ( 1 )
  {
    v11 = sub_2255020(a1, v7, v8);
    if ( v11 )
      break;
    v12 = &v7[strlen(v7)];
    v13 = &v8[strlen(v8)];
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
  if ( v17 - 24 != (char *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v15 = _InterlockedExchangeAdd((volatile signed __int32 *)v17 - 2, 0xFFFFFFFF);
    }
    else
    {
      v15 = *((_DWORD *)v17 - 2);
      *((_DWORD *)v17 - 2) = v15 - 1;
    }
    if ( v15 <= 0 )
      j_j___libc_free_0_1((unsigned __int64)(v17 - 24));
  }
  if ( s - 24 != (char *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v16 = _InterlockedExchangeAdd((volatile signed __int32 *)s - 2, 0xFFFFFFFF);
    }
    else
    {
      v16 = *((_DWORD *)s - 2);
      *((_DWORD *)s - 2) = v16 - 1;
    }
    if ( v16 <= 0 )
      j_j___libc_free_0_1((unsigned __int64)(s - 24));
  }
  return v11;
}
