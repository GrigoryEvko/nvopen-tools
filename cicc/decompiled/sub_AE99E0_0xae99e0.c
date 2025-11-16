// Function: sub_AE99E0
// Address: 0xae99e0
//
__int64 __fastcall sub_AE99E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rsi
  _QWORD *v7; // r13
  _QWORD *i; // r15
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 *v13; // rbx
  __int64 *v14; // r12
  __int64 v15; // rdi
  __int64 *v16; // r12
  __int64 *v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  _QWORD *v23; // [rsp+8h] [rbp-138h]
  __int64 v24; // [rsp+20h] [rbp-120h]
  __int64 v25; // [rsp+28h] [rbp-118h]
  __int64 *v26; // [rsp+30h] [rbp-110h] BYREF
  __int64 v27; // [rsp+38h] [rbp-108h]
  _BYTE v28[96]; // [rsp+40h] [rbp-100h] BYREF
  __int64 *v29; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v30; // [rsp+A8h] [rbp-98h]
  _BYTE v31[144]; // [rsp+B0h] [rbp-90h] BYREF

  v4 = 0xC00000000LL;
  result = a1 + 72;
  v6 = (__int64)v28;
  v7 = *(_QWORD **)(a1 + 80);
  v26 = (__int64 *)v28;
  v27 = 0xC00000000LL;
  v29 = (__int64 *)v31;
  v30 = 0xC00000000LL;
  v23 = (_QWORD *)(a1 + 72);
  if ( v7 != (_QWORD *)(a1 + 72) )
  {
    do
    {
      if ( !v7 )
        BUG();
      for ( i = (_QWORD *)v7[4]; v7 + 3 != i; i = (_QWORD *)i[1] )
      {
        if ( !i )
          BUG();
        v9 = i[5];
        if ( v9 )
        {
          v10 = sub_B14240(v9);
          v12 = v11;
          if ( v11 != v10 )
          {
            while ( *(_BYTE *)(v10 + 32) )
            {
              v10 = *(_QWORD *)(v10 + 8);
              if ( v11 == v10 )
                goto LABEL_15;
            }
LABEL_10:
            if ( v12 != v10 )
            {
              if ( *(_BYTE *)(v10 + 64) == 2 )
              {
                v19 = (unsigned int)v30;
                if ( (unsigned __int64)(unsigned int)v30 + 1 > HIDWORD(v30) )
                {
                  v24 = v12;
                  v25 = v10;
                  sub_C8D5F0(&v29, v31, (unsigned int)v30 + 1LL, 8);
                  v19 = (unsigned int)v30;
                  v12 = v24;
                  v10 = v25;
                }
                v29[v19] = v10;
                LODWORD(v30) = v30 + 1;
              }
              while ( 1 )
              {
                v10 = *(_QWORD *)(v10 + 8);
                if ( v12 == v10 )
                  break;
                if ( !*(_BYTE *)(v10 + 32) )
                  goto LABEL_10;
              }
            }
          }
        }
LABEL_15:
        if ( *((_BYTE *)i - 24) == 85
          && (v20 = *(i - 7)) != 0
          && !*(_BYTE *)v20
          && (v6 = i[7], *(_QWORD *)(v20 + 24) == v6)
          && (*(_BYTE *)(v20 + 33) & 0x20) != 0
          && *(_DWORD *)(v20 + 36) == 68 )
        {
          v21 = (unsigned int)v27;
          a4 = HIDWORD(v27);
          v22 = (unsigned int)v27 + 1LL;
          if ( v22 > HIDWORD(v27) )
          {
            v6 = (__int64)v28;
            sub_C8D5F0(&v26, v28, v22, 8);
            v21 = (unsigned int)v27;
          }
          v4 = (__int64)v26;
          v26[v21] = (__int64)(i - 3);
          LODWORD(v27) = v27 + 1;
        }
        else
        {
          v6 = 38;
          sub_B99FD0(i - 3, 38, 0);
        }
      }
      v7 = (_QWORD *)v7[1];
    }
    while ( v23 != v7 );
    v13 = v26;
    v14 = &v26[(unsigned int)v27];
    if ( v14 != v26 )
    {
      do
      {
        v15 = *v13++;
        sub_B43D60(v15, v6, v4, a4);
      }
      while ( v14 != v13 );
    }
    result = (__int64)v29;
    v16 = &v29[(unsigned int)v30];
    if ( v29 != v16 )
    {
      v17 = v29;
      do
      {
        v18 = *v17++;
        result = sub_B14290(v18);
      }
      while ( v16 != v17 );
      v16 = v29;
    }
    if ( v16 != (__int64 *)v31 )
      result = _libc_free(v16, v6);
  }
  if ( v26 != (__int64 *)v28 )
    return _libc_free(v26, v6);
  return result;
}
