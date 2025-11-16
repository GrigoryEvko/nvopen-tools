// Function: sub_2AAF230
// Address: 0x2aaf230
//
void __fastcall sub_2AAF230(_QWORD *a1, void **a2)
{
  _QWORD *v3; // rdi
  size_t v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rsi
  size_t v7; // rdx
  _QWORD *v8; // [rsp+0h] [rbp-30h] BYREF
  size_t n; // [rsp+8h] [rbp-28h]
  _QWORD src[4]; // [rsp+10h] [rbp-20h] BYREF

  sub_CA0F50((__int64 *)&v8, a2);
  v3 = (_QWORD *)a1[2];
  if ( v8 == src )
  {
    v7 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v3 = src[0];
      else
        memcpy(v3, src, n);
      v7 = n;
      v3 = (_QWORD *)a1[2];
    }
    a1[3] = v7;
    *((_BYTE *)v3 + v7) = 0;
    v3 = v8;
  }
  else
  {
    v4 = n;
    v5 = src[0];
    if ( v3 == a1 + 4 )
    {
      a1[2] = v8;
      a1[3] = v4;
      a1[4] = v5;
    }
    else
    {
      v6 = a1[4];
      a1[2] = v8;
      a1[3] = v4;
      a1[4] = v5;
      if ( v3 )
      {
        v8 = v3;
        src[0] = v6;
        goto LABEL_5;
      }
    }
    v8 = src;
    v3 = src;
  }
LABEL_5:
  n = 0;
  *(_BYTE *)v3 = 0;
  if ( v8 != src )
    j_j___libc_free_0((unsigned __int64)v8);
}
