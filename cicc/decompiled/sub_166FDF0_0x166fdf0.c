// Function: sub_166FDF0
// Address: 0x166fdf0
//
void __fastcall sub_166FDF0(__int64 a1, void *a2, size_t a3)
{
  char v3; // al
  const char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r13
  void *s2; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  __int64 v10[2]; // [rsp+10h] [rbp-30h] BYREF
  __int16 v11; // [rsp+20h] [rbp-20h]

  v3 = *(_BYTE *)(a1 + 32);
  s2 = a2;
  n = a3;
  if ( (v3 & 0xFu) - 7 > 1 )
  {
    v4 = sub_1649960(a1);
    if ( n != v5 || n && memcmp(v4, s2, n) )
    {
      v6 = sub_1632000(*(_QWORD *)(a1 + 40), (__int64)s2, n);
      v7 = v6;
      if ( v6 )
      {
        sub_164B7C0(a1, v6);
        v10[0] = (__int64)&s2;
        v11 = 261;
        sub_164B780(v7, v10);
      }
      else
      {
        v11 = 261;
        v10[0] = (__int64)&s2;
        sub_164B780(a1, v10);
      }
    }
  }
}
