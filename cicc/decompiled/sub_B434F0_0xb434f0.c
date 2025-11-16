// Function: sub_B434F0
// Address: 0xb434f0
//
__int64 __fastcall sub_B434F0(__int64 a1)
{
  char *v2; // rsi
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rbx
  _QWORD *v14; // r15
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  _QWORD *v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // r15
  _QWORD *v23; // r14
  _QWORD *v24; // r13
  _QWORD *v25; // rbx
  __int64 v26; // [rsp+0h] [rbp-350h]
  unsigned __int8 v27; // [rsp+8h] [rbp-348h]
  __int64 v28; // [rsp+8h] [rbp-348h]
  __int64 v29; // [rsp+10h] [rbp-340h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-338h]
  _BYTE v31[816]; // [rsp+20h] [rbp-330h] BYREF

  v27 = *(_BYTE *)(a1 + 96);
  if ( !v27 )
  {
    v2 = *(char **)(a1 + 56);
    sub_B428A0(&v29, v2, *(_QWORD *)(a1 + 64));
    v3 = v29;
    v4 = 192LL * v30;
    v5 = v29 + v4;
    if ( v29 + v4 == v29 )
    {
LABEL_58:
      if ( (_BYTE *)v3 != v31 )
        _libc_free(v3, v2);
    }
    else
    {
      v6 = v29;
      while ( 1 )
      {
        if ( *(_DWORD *)v6 == 2 )
        {
          v7 = *(_QWORD *)(v6 + 16);
          v8 = v7 + 32LL * *(unsigned int *)(v6 + 24);
          if ( v8 != v7 )
            break;
        }
LABEL_5:
        v6 += 192;
        if ( v5 == v6 )
        {
          v26 = v29;
          v3 = v29 + 192LL * v30;
          if ( v29 != v3 )
          {
            do
            {
              v18 = *(unsigned int *)(v3 - 120);
              v19 = *(_QWORD *)(v3 - 128);
              v3 -= 192;
              v20 = v19 + 56 * v18;
              if ( v19 != v20 )
              {
                do
                {
                  v21 = *(unsigned int *)(v20 - 40);
                  v22 = *(_QWORD **)(v20 - 48);
                  v20 -= 56;
                  v21 *= 32;
                  v23 = (_QWORD *)((char *)v22 + v21);
                  if ( v22 != (_QWORD *)((char *)v22 + v21) )
                  {
                    do
                    {
                      v23 -= 4;
                      if ( (_QWORD *)*v23 != v23 + 2 )
                      {
                        v2 = (char *)(v23[2] + 1LL);
                        j_j___libc_free_0(*v23, v2);
                      }
                    }
                    while ( v22 != v23 );
                    v22 = *(_QWORD **)(v20 + 8);
                  }
                  if ( v22 != (_QWORD *)(v20 + 24) )
                    _libc_free(v22, v2);
                }
                while ( v19 != v20 );
                v19 = *(_QWORD *)(v3 + 64);
              }
              if ( v19 != v3 + 80 )
                _libc_free(v19, v2);
              v24 = *(_QWORD **)(v3 + 16);
              v25 = &v24[4 * *(unsigned int *)(v3 + 24)];
              if ( v24 != v25 )
              {
                do
                {
                  v25 -= 4;
                  if ( (_QWORD *)*v25 != v25 + 2 )
                  {
                    v2 = (char *)(v25[2] + 1LL);
                    j_j___libc_free_0(*v25, v2);
                  }
                }
                while ( v24 != v25 );
                v24 = *(_QWORD **)(v3 + 16);
              }
              if ( v24 != (_QWORD *)(v3 + 32) )
                _libc_free(v24, v2);
            }
            while ( v26 != v3 );
            v3 = v29;
          }
          goto LABEL_58;
        }
      }
      while ( 1 )
      {
        v2 = "{memory}";
        if ( !(unsigned int)sub_2241AC0(v7, "{memory}") )
          break;
        v7 += 32;
        if ( v8 == v7 )
          goto LABEL_5;
      }
      v28 = v29;
      v9 = v29 + 192LL * v30;
      if ( v29 != v9 )
      {
        do
        {
          v10 = *(unsigned int *)(v9 - 120);
          v11 = *(_QWORD *)(v9 - 128);
          v9 -= 192;
          v12 = v11 + 56 * v10;
          if ( v11 != v12 )
          {
            do
            {
              v13 = *(unsigned int *)(v12 - 40);
              v14 = *(_QWORD **)(v12 - 48);
              v12 -= 56;
              v15 = &v14[4 * v13];
              if ( v14 != v15 )
              {
                do
                {
                  v15 -= 4;
                  if ( (_QWORD *)*v15 != v15 + 2 )
                  {
                    v2 = (char *)(v15[2] + 1LL);
                    j_j___libc_free_0(*v15, v2);
                  }
                }
                while ( v14 != v15 );
                v14 = *(_QWORD **)(v12 + 8);
              }
              if ( v14 != (_QWORD *)(v12 + 24) )
                _libc_free(v14, v2);
            }
            while ( v11 != v12 );
            v11 = *(_QWORD *)(v9 + 64);
          }
          if ( v11 != v9 + 80 )
            _libc_free(v11, v2);
          v16 = *(_QWORD **)(v9 + 16);
          v17 = &v16[4 * *(unsigned int *)(v9 + 24)];
          if ( v16 != v17 )
          {
            do
            {
              v17 -= 4;
              if ( (_QWORD *)*v17 != v17 + 2 )
              {
                v2 = (char *)(v17[2] + 1LL);
                j_j___libc_free_0(*v17, v2);
              }
            }
            while ( v16 != v17 );
            v16 = *(_QWORD **)(v9 + 16);
          }
          if ( v16 != (_QWORD *)(v9 + 32) )
            _libc_free(v16, v2);
        }
        while ( v28 != v9 );
        v9 = v29;
      }
      if ( (_BYTE *)v9 != v31 )
        _libc_free(v9, v2);
      return 1;
    }
  }
  return v27;
}
