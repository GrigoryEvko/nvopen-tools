// Function: sub_3145F20
// Address: 0x3145f20
//
__int64 __fastcall sub_3145F20(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  _QWORD *v4; // rdi
  unsigned __int64 v5; // rsi
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  _QWORD *v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v10; // [rsp+8h] [rbp-28h]
  _QWORD *v11; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v12; // [rsp+18h] [rbp-18h]

  v9 = (_QWORD *)a1;
  v10 = a2;
  v2 = sub_C93460((__int64 *)&v9, ".content.", 9u);
  if ( v2 == -1 || (v3 = v2 + 9, v3 > v10) || (v4 = (_QWORD *)((char *)v9 + v3), v5 = v10 - v3, v10 == v3) )
  {
    v7 = sub_C93460((__int64 *)&v9, ".llvm.", 6u);
    if ( v7 == -1 )
    {
      v7 = v10;
    }
    else if ( v10 <= v7 )
    {
      v7 = v10;
    }
    v11 = v9;
    v12 = v7;
    v8 = sub_C93460((__int64 *)&v11, ".__uniq.", 8u);
    v4 = v11;
    v5 = v8;
    if ( v8 == -1 )
    {
      v5 = v12;
    }
    else if ( v12 <= v8 )
    {
      v5 = v12;
    }
  }
  return sub_CBF760(v4, v5);
}
