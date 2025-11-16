// Function: sub_2B31420
// Address: 0x2b31420
//
void __fastcall sub_2B31420(unsigned int a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // r9
  __int64 v5; // r14
  unsigned int v6; // ebx
  unsigned int v7; // eax
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned __int64 v11; // [rsp+0h] [rbp-80h]
  void *s; // [rsp+10h] [rbp-70h] BYREF
  __int64 v13; // [rsp+18h] [rbp-68h]
  _DWORD v14[24]; // [rsp+20h] [rbp-60h] BYREF

  v4 = a4;
  v5 = a3;
  s = v14;
  v6 = *(_DWORD *)(a2 + 8);
  v13 = 0xC00000000LL;
  if ( a4 > 0xC )
  {
    v11 = a4;
    sub_C8D5F0((__int64)&s, v14, a4, 4u, (__int64)&s, a4);
    v4 = v11;
    a3 = 4 * v11;
    if ( 4 * v11 )
    {
      memset(s, 255, a3);
      v4 = v11;
    }
  }
  else if ( a4 )
  {
    v8 = 4 * a4;
    if ( 4 * a4 )
    {
      if ( v8 < 8 )
      {
        if ( (v8 & 4) != 0 )
        {
          v14[0] = -1;
          v14[v8 / 4 - 1] = -1;
        }
        else if ( v8 )
        {
          LOBYTE(v14[0]) = -1;
        }
      }
      else
      {
        a3 = v8;
        v9 = v8 - 1;
        *(_QWORD *)((char *)&v14[-2] + a3) = -1;
        if ( v9 >= 8 )
        {
          v10 = v9 & 0xFFFFFFF8;
          LODWORD(a3) = 0;
          do
          {
            a4 = (unsigned int)a3;
            a3 = (unsigned int)(a3 + 8);
            *(_QWORD *)((char *)v14 + a4) = -1;
          }
          while ( (unsigned int)a3 < v10 );
        }
      }
    }
  }
  LODWORD(v13) = v4;
  if ( (int)v4 > 0 )
  {
    a4 = 0;
    do
    {
      v7 = *(_DWORD *)(v5 + a4);
      if ( v7 != -1 )
      {
        a3 = *(unsigned int *)(*(_QWORD *)a2 + 4LL * (v7 % v6));
        if ( (_DWORD)a3 != -1 )
          a3 = (unsigned int)a3 % a1;
        *(_DWORD *)((char *)s + a4) = a3;
      }
      a4 += 4LL;
    }
    while ( a4 != 4LL * (unsigned int)(v4 - 1) + 4 );
  }
  sub_2B310D0(a2, (__int64)&s, a3, a4, (__int64)&s, v4);
  if ( s != v14 )
    _libc_free((unsigned __int64)s);
}
