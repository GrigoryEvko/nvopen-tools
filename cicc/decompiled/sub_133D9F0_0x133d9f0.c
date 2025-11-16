// Function: sub_133D9F0
// Address: 0x133d9f0
//
__int64 __fastcall sub_133D9F0(__int64 *a1, __int64 *a2, unsigned __int64 a3)
{
  int v4; // r8d
  __int64 result; // rax
  unsigned __int64 v6; // r13
  size_t v7; // r14
  __int64 *v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rax
  bool v14; // cc
  unsigned __int64 v15; // rcx
  __int64 i; // rax
  unsigned __int64 v17; // rcx
  __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( !(unsigned __int8)sub_130B0B0() && (int)sub_130B150(a1 + 17, a2) > 0 )
  {
    sub_130B140(a1 + 17, a2);
    sub_133D760(a1);
  }
  v4 = sub_130B150(a1 + 19, a2);
  result = 0;
  if ( v4 <= 0 )
  {
    sub_130B140(v18, a2);
    sub_130B1F0(v18, a1 + 17);
    v6 = sub_130B220(v18, a1 + 16);
    sub_130B140(v18, a1 + 16);
    sub_130B200(v18, v6);
    sub_130B1D0(a1 + 17, v18);
    sub_133D760(a1);
    if ( v6 > 0xC7 )
    {
      v10 = 0;
      a1[22] = 0;
      a1[220] = 0;
      memset(
        (void *)((unsigned __int64)(a1 + 23) & 0xFFFFFFFFFFFFFFF8LL),
        0,
        8LL * (((_DWORD)a1 + 176 - (((_DWORD)a1 + 184) & 0xFFFFFFF8) + 1592) >> 3));
    }
    else
    {
      v7 = 200 - v6;
      memmove(a1 + 22, &a1[v6 + 22], v7 * 8);
      if ( v6 > 1 )
      {
        v8 = &a1[v7 + 22];
        v9 = (unsigned int)(8 * v6 - 8);
        *v8 = 0;
        *(__int64 *)((char *)v8 + v9 - 8) = 0;
        memset(
          (void *)((unsigned __int64)&a1[v7 + 23] & 0xFFFFFFFFFFFFFFF8LL),
          0,
          8LL * (((unsigned int)v9 + (_DWORD)v8 - (((_DWORD)a1 + (_DWORD)(v7 * 8) + 184) & 0xFFFFFFF8)) >> 3));
      }
      v10 = a1[22];
    }
    v11 = a1[21];
    v12 = 20;
    v13 = a3 - v11;
    v14 = a3 <= v11;
    v15 = 0;
    if ( v14 )
      v13 = 0;
    a1[221] = v13;
    for ( i = 0; ; v12 = qword_42878A0[i] )
    {
      ++i;
      v15 += v12 * v10;
      if ( i == 200 )
        break;
      v10 = a1[i + 22];
    }
    v17 = v15 >> 24;
    result = 1;
    a1[20] = v17;
    if ( a3 >= v17 )
      v17 = a3;
    a1[21] = v17;
  }
  return result;
}
