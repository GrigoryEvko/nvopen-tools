// Function: sub_152FD80
// Address: 0x152fd80
//
void __fastcall sub_152FD80(_DWORD **a1)
{
  char *v2; // rdx
  char *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  unsigned int v6; // r15d
  int v7; // eax
  int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  _DWORD *v12; // rsi
  int v13; // r10d
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  int i; // eax
  _DWORD *v18; // rdi
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // [rsp-270h] [rbp-270h]
  __int64 v23; // [rsp-268h] [rbp-268h]
  __int64 v24; // [rsp-250h] [rbp-250h] BYREF
  _BYTE *v25; // [rsp-248h] [rbp-248h] BYREF
  __int64 v26; // [rsp-240h] [rbp-240h]
  _BYTE v27[568]; // [rsp-238h] [rbp-238h] BYREF

  if ( a1[55] != a1[56] )
  {
    sub_1526BE0(*a1, 9u, 3u);
    v2 = (char *)a1[56];
    v25 = v27;
    v26 = 0x4000000000LL;
    v3 = (char *)a1[55];
    v4 = (v2 - v3) >> 3;
    if ( (_DWORD)v4 )
    {
      v5 = 0;
      v23 = 8LL * (unsigned int)(v4 - 1);
      while ( 1 )
      {
        v6 = -1;
        v24 = *(_QWORD *)&v3[v5];
        v7 = sub_15601D0(&v24);
        v8 = v7 - 1;
        if ( v7 )
        {
          do
          {
            v9 = sub_15601E0(&v24, v6);
            v10 = v9;
            if ( v9 )
            {
              v11 = *((unsigned int *)a1 + 94);
              v12 = a1[45];
              if ( (_DWORD)v11 )
              {
                v13 = 1;
                v14 = (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4) | ((unsigned __int64)(37 * v6) << 32))
                    - 1
                    - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32);
                v15 = ((v14 >> 22) ^ v14) - 1 - (((v14 >> 22) ^ v14) << 13);
                v16 = ((9 * ((v15 >> 8) ^ v15)) >> 15) ^ (9 * ((v15 >> 8) ^ v15));
                for ( i = (v11 - 1) & (((v16 - 1 - (v16 << 27)) >> 31) ^ (v16 - 1 - ((_DWORD)v16 << 27)));
                      ;
                      i = (v11 - 1) & v19 )
                {
                  v18 = &v12[6 * i];
                  if ( v6 == *v18 && v10 == *((_QWORD *)v18 + 1) )
                    break;
                  if ( *v18 == -1 && *((_QWORD *)v18 + 1) == -4 )
                    goto LABEL_12;
                  v19 = v13 + i;
                  ++v13;
                }
              }
              else
              {
LABEL_12:
                v18 = &v12[6 * v11];
              }
              v20 = (unsigned int)v18[4];
              v21 = (unsigned int)v26;
              if ( (unsigned int)v26 >= HIDWORD(v26) )
              {
                v22 = (unsigned int)v18[4];
                sub_16CD150(&v25, v27, 0, 8);
                v21 = (unsigned int)v26;
                v20 = v22;
              }
              *(_QWORD *)&v25[8 * v21] = v20;
              LODWORD(v26) = v26 + 1;
            }
            ++v6;
          }
          while ( v6 != v8 );
        }
        sub_152F3D0(*a1, 2u, (__int64)&v25, 0);
        LODWORD(v26) = 0;
        if ( v5 == v23 )
          break;
        v3 = (char *)a1[55];
        v5 += 8;
      }
    }
    sub_15263C0((__int64 **)*a1);
    if ( v25 != v27 )
      _libc_free((unsigned __int64)v25);
  }
}
