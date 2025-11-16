// Function: sub_823420
// Address: 0x823420
//
void *__fastcall sub_823420(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD **v3; // rcx
  unsigned __int64 v4; // r8
  _QWORD *v5; // rdi
  _QWORD *v6; // rax
  _QWORD *v7; // rcx
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 *v11; // rax
  _QWORD *v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rbx
  _QWORD *v15; // r12
  _QWORD *v16; // r14
  _QWORD *v17; // rdi
  int i; // ebx
  void *result; // rax
  _QWORD *v20; // r12
  _QWORD *v21; // rbx
  _QWORD *v22; // rdi
  _QWORD *v23; // rbx
  _QWORD *v24; // rdi
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

  if ( dword_4F195D8 )
  {
    v2 = qword_4F195D0;
    if ( qword_4F195D0 )
    {
      sub_822B90(*(_QWORD *)qword_4F195D0, 16LL * (unsigned int)(*(_DWORD *)(qword_4F195D0 + 8) + 1));
      a2 = 16;
      sub_822B90(v2, 16);
      qword_4F195D0 = 0;
    }
    v3 = (_QWORD **)qword_4F073B0;
    if ( dword_4F073A8 )
    {
      a2 = 8LL * dword_4F073A8;
      v4 = 8 * (dword_4F073A8 - (unsigned __int64)(unsigned int)(dword_4F073A8 - 1));
      while ( 1 )
      {
        v5 = (_QWORD **)((char *)v3 + a2);
        v6 = *(_QWORD **)((char *)v3 + a2);
        if ( v6 )
        {
          v7 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              v8 = (_QWORD *)*v6;
              if ( !v6[4] )
                break;
              v7 = v6;
              v6 = (_QWORD *)*v6;
              if ( !v8 )
                goto LABEL_13;
            }
            if ( !v7 )
              break;
            *v7 = v8;
            v6 = v8;
LABEL_10:
            if ( !v6 )
              goto LABEL_13;
          }
          while ( 1 )
          {
            *v5 = v8;
            if ( !v8 )
              break;
            v6 = (_QWORD *)*v8;
            if ( v8[4] )
            {
              v7 = v8;
              goto LABEL_10;
            }
            v8 = (_QWORD *)*v8;
          }
LABEL_13:
          v3 = (_QWORD **)qword_4F073B0;
        }
        if ( a2 == v4 )
          break;
        a2 -= 8;
      }
    }
    v9 = *v3;
    if ( *v3 )
    {
      a2 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v10 = (_QWORD *)*v9;
          if ( !v9[4] )
            break;
          a2 = (__int64)v9;
          v9 = (_QWORD *)*v9;
          if ( !v10 )
            goto LABEL_27;
        }
        if ( !a2 )
          break;
        *(_QWORD *)a2 = v10;
        v9 = v10;
LABEL_24:
        if ( !v9 )
          goto LABEL_27;
      }
      while ( 1 )
      {
        *v3 = v10;
        if ( !v10 )
          break;
        v9 = (_QWORD *)*v10;
        if ( v10[4] )
        {
          a2 = (__int64)v10;
          goto LABEL_24;
        }
        v10 = (_QWORD *)*v10;
      }
    }
LABEL_27:
    v11 = (__int64 *)qword_4F195E0;
    if ( qword_4F195E0 )
    {
      v12 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v13 = *v11;
          if ( !v11[4] )
            break;
          v12 = v11;
          v11 = (__int64 *)*v11;
          if ( !v13 )
            goto LABEL_34;
        }
        if ( !v12 )
          break;
        *v12 = v13;
        v11 = (__int64 *)v13;
LABEL_31:
        if ( !v11 )
          goto LABEL_34;
      }
      while ( 1 )
      {
        qword_4F195E0 = v13;
        if ( !v13 )
          break;
        v11 = *(__int64 **)v13;
        if ( *(_QWORD *)(v13 + 32) )
        {
          v12 = (_QWORD *)v13;
          goto LABEL_31;
        }
        v13 = *(_QWORD *)v13;
      }
    }
LABEL_34:
    if ( dword_4F073A8 )
    {
      v14 = 8LL * dword_4F073A8;
      v25 = 8 * (dword_4F073A8 - (unsigned __int64)(unsigned int)(dword_4F073A8 - 1));
      while ( 1 )
      {
        v15 = (char *)qword_4F073B0 + v14;
        v16 = *(_QWORD **)((char *)qword_4F073B0 + v14);
        while ( v16 )
        {
          v17 = v16;
          v16 = (_QWORD *)*v16;
          _libc_free(v17, a2);
        }
        *v15 = 0;
        *(_QWORD *)((char *)qword_4F072B0 + v14) = 0;
        if ( v25 == v14 )
          break;
        v14 -= 8;
      }
    }
    v20 = qword_4F073B0;
    v21 = *(_QWORD **)qword_4F073B0;
    if ( *(_QWORD *)qword_4F073B0 )
    {
      do
      {
        v22 = v21;
        v21 = (_QWORD *)*v21;
        _libc_free(v22, a2);
      }
      while ( v21 );
    }
    *v20 = 0;
    result = qword_4F072B0;
    *(_QWORD *)qword_4F072B0 = 0;
    v23 = (_QWORD *)qword_4F195E0;
    if ( qword_4F195E0 )
    {
      do
      {
        v24 = v23;
        v23 = (_QWORD *)*v23;
        result = (void *)_libc_free(v24, a2);
      }
      while ( v23 );
    }
  }
  else
  {
    for ( i = dword_4F073A8; i; --i )
      sub_823310(i, a2);
    sub_823310(0, a2);
    result = sub_822A90();
  }
  qword_4F195E0 = 0;
  return result;
}
