// Function: sub_1254A40
// Address: 0x1254a40
//
unsigned __int64 *__fastcall sub_1254A40(unsigned __int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // r14
  _BYTE *v8; // [rsp+8h] [rbp-68h]
  __int64 i; // [rsp+18h] [rbp-58h]
  unsigned __int64 v11; // [rsp+28h] [rbp-48h] BYREF
  void *s; // [rsp+30h] [rbp-40h] BYREF
  size_t n; // [rsp+38h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 56);
  for ( i = v3; ; i = *(_QWORD *)(a2 + 56) )
  {
    s = 0;
    n = 0;
    sub_12548E0(&v11, a2, (__int64)&s);
    v4 = v11 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      break;
    if ( n )
    {
      v8 = s;
      v5 = memchr(s, 0, n);
      if ( v5 )
      {
        *(_QWORD *)(a2 + 56) = v3;
        v6 = v5 - v8 + i;
        sub_12549C0((unsigned __int64 *)&s, a2, a3, v6 - v3);
        v4 = (unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL;
        if ( ((unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL) == 0 )
        {
          *(_QWORD *)(a2 + 56) = v6 + 1;
          *a1 = 1;
          return a1;
        }
        break;
      }
    }
  }
  *a1 = v4 | 1;
  return a1;
}
