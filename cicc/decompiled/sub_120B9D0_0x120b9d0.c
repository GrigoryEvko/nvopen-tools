// Function: sub_120B9D0
// Address: 0x120b9d0
//
__int64 __fastcall sub_120B9D0(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v3; // rcx
  _QWORD *v4; // rbx
  __int64 v5; // r14
  _QWORD *v6; // rax
  unsigned __int64 v7; // r9
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v11; // [rsp+18h] [rbp-48h]
  _QWORD v12[8]; // [rsp+20h] [rbp-40h] BYREF

  v10 = v12;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v11 = 0;
  LOBYTE(v12[0]) = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 101, "expected 'module asm'")
    || (v1 = sub_120B3D0(a1, (__int64)&v10), (_BYTE)v1) )
  {
    v1 = 1;
  }
  else
  {
    v4 = *(_QWORD **)(a1 + 344);
    if ( v11 > 0x3FFFFFFFFFFFFFFFLL - v4[12] )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(v4 + 11, v10, v11, v3);
    v5 = v4[12];
    if ( v5 )
    {
      v6 = (_QWORD *)v4[11];
      if ( *((_BYTE *)v6 + v5 - 1) != 10 )
      {
        v7 = v5 + 1;
        if ( v6 == v4 + 13 )
          v8 = 15;
        else
          v8 = v4[13];
        if ( v7 > v8 )
        {
          sub_2240BB0(v4 + 11, v4[12], 0, 0, 1);
          v6 = (_QWORD *)v4[11];
          v7 = v5 + 1;
        }
        *((_BYTE *)v6 + v5) = 10;
        v9 = v4[11];
        v4[12] = v7;
        *(_BYTE *)(v9 + v5 + 1) = 0;
      }
    }
  }
  if ( v10 != v12 )
    j_j___libc_free_0(v10, v12[0] + 1LL);
  return v1;
}
