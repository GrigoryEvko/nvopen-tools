// Function: sub_AE9860
// Address: 0xae9860
//
__int64 __fastcall sub_AE9860(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 *v10; // rdi
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // r15
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 *v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h]
  _BYTE v17[80]; // [rsp+20h] [rbp-50h] BYREF

  result = 0x400000000LL;
  v5 = (_QWORD *)(a2 + 8 * a3);
  v15 = (__int64 *)v17;
  v16 = 0x400000000LL;
  if ( v5 == (_QWORD *)a2 )
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
      return result;
  }
  else
  {
    v6 = (_QWORD *)a2;
    do
    {
      if ( (*(_BYTE *)(*v6 + 7LL) & 0x20) != 0 )
      {
        a2 = 38;
        v7 = sub_B91C10(*v6, 38);
        if ( v7 )
        {
          v8 = (unsigned int)v16;
          if ( (unsigned __int64)(unsigned int)v16 + 1 > HIDWORD(v16) )
          {
            a2 = (__int64)v17;
            v14 = v7;
            sub_C8D5F0(&v15, v17, (unsigned int)v16 + 1LL, 8);
            v8 = (unsigned int)v16;
            v7 = v14;
          }
          v15[v8] = v7;
          LODWORD(v16) = v16 + 1;
        }
      }
      ++v6;
    }
    while ( v5 != v6 );
    result = (unsigned int)v16;
    if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
      goto LABEL_14;
  }
  a2 = 38;
  v9 = sub_B91C10(a1, 38);
  result = (unsigned int)v16;
  if ( v9 )
  {
    if ( (unsigned __int64)(unsigned int)v16 + 1 > HIDWORD(v16) )
    {
      a2 = (__int64)v17;
      sub_C8D5F0(&v15, v17, (unsigned int)v16 + 1LL, 8);
      result = (unsigned int)v16;
    }
    v15[result] = v9;
    result = (unsigned int)(v16 + 1);
    LODWORD(v16) = v16 + 1;
  }
LABEL_14:
  v10 = v15;
  if ( (_DWORD)result )
  {
    v11 = v15 + 1;
    v12 = &v15[result];
    v13 = *v15;
    if ( v12 != v15 + 1 )
    {
      do
      {
        if ( *v11 != v13 )
          sub_AE9740(*v11, v13);
        ++v11;
      }
      while ( v11 != v12 );
    }
    a2 = 38;
    result = sub_B99FD0(a1, 38, v13);
    v10 = v15;
  }
  if ( v10 != (__int64 *)v17 )
    return _libc_free(v10, a2);
  return result;
}
