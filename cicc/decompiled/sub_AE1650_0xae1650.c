// Function: sub_AE1650
// Address: 0xae1650
//
_QWORD *__fastcall sub_AE1650(_QWORD *a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // r13
  unsigned int v8; // ebx
  __int64 v9; // rdi
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // r14
  unsigned int v16; // ebx
  __int64 v17[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v18[8]; // [rsp+10h] [rbp-40h] BYREF

  if ( a3 )
  {
    if ( !(unsigned __int8)sub_C93C90(a2, a3, 10, v17) )
    {
      v12 = v17[0];
      if ( v17[0] == LODWORD(v17[0]) )
      {
        *a4 = v17[0];
        if ( v12 <= 0xFFFFFF )
        {
          *a1 = 1;
          return a1;
        }
      }
    }
    v13 = sub_C63BB0();
    v15 = v14;
    v16 = v13;
    v17[0] = (__int64)v18;
    sub_AE11D0(v17, "address space must be a 24-bit integer", (__int64)"");
    sub_C63F00(a1, v17, v16, v15);
    v9 = v17[0];
    if ( (_QWORD *)v17[0] != v18 )
      goto LABEL_3;
  }
  else
  {
    v5 = sub_C63BB0();
    v7 = v6;
    v8 = v5;
    v17[0] = (__int64)v18;
    sub_AE11D0(v17, "address space component cannot be empty", (__int64)"");
    sub_C63F00(a1, v17, v8, v7);
    v9 = v17[0];
    if ( (_QWORD *)v17[0] != v18 )
LABEL_3:
      j_j___libc_free_0(v9, v18[0] + 1LL);
  }
  return a1;
}
