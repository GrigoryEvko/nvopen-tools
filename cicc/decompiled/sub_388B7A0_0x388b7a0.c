// Function: sub_388B7A0
// Address: 0x388b7a0
//
__int64 __fastcall sub_388B7A0(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int64 *v3; // rbx
  unsigned __int64 v4; // r14
  unsigned __int64 *v5; // rax
  unsigned __int64 v6; // r9
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  char *v9; // [rsp+10h] [rbp-50h] BYREF
  size_t v10; // [rsp+18h] [rbp-48h]
  _BYTE v11[64]; // [rsp+20h] [rbp-40h] BYREF

  v9 = v11;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v10 = 0;
  v11[0] = 0;
  if ( (unsigned __int8)sub_388AF10(a1, 94, "expected 'module asm'")
    || (v1 = sub_388B0A0(a1, (unsigned __int64 *)&v9), (_BYTE)v1) )
  {
    v1 = 1;
  }
  else
  {
    v3 = *(unsigned __int64 **)(a1 + 176);
    if ( v10 > 0x3FFFFFFFFFFFFFFFLL - v3[12] )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(v3 + 11, v9, v10);
    v4 = v3[12];
    if ( v4 )
    {
      v5 = (unsigned __int64 *)v3[11];
      if ( *((_BYTE *)v5 + v4 - 1) != 10 )
      {
        v6 = v4 + 1;
        if ( v5 == v3 + 13 )
          v7 = 15;
        else
          v7 = v3[13];
        if ( v6 > v7 )
        {
          sub_2240BB0(v3 + 11, v3[12], 0, 0, 1u);
          v5 = (unsigned __int64 *)v3[11];
          v6 = v4 + 1;
        }
        *((_BYTE *)v5 + v4) = 10;
        v8 = v3[11];
        v3[12] = v6;
        *(_BYTE *)(v8 + v4 + 1) = 0;
      }
    }
  }
  if ( v9 != v11 )
    j_j___libc_free_0((unsigned __int64)v9);
  return v1;
}
