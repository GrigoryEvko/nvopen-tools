// Function: sub_AE1770
// Address: 0xae1770
//
_QWORD *__fastcall sub_AE1770(_QWORD *a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r13
  unsigned int v11; // ebx
  int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // rdx
  _QWORD v17[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+20h] [rbp-60h] BYREF
  __int64 v20; // [rsp+28h] [rbp-58h]
  const char *v21; // [rsp+30h] [rbp-50h]
  __int16 v22; // [rsp+40h] [rbp-40h]

  if ( a3 )
  {
    if ( !(unsigned __int8)sub_C93C90(a2, a3, 10, &v19) )
    {
      v14 = v19;
      if ( v19 == (unsigned int)v19 )
      {
        *a4 = v19;
        if ( (unsigned int)(v14 - 1) <= 0xFFFFFE )
        {
          *a1 = 1;
          return a1;
        }
      }
    }
    v19 = a5;
    v22 = 773;
    v20 = a6;
    v21 = " must be a non-zero 24-bit integer";
    v15 = sub_C63BB0();
    v10 = v16;
    v11 = v15;
  }
  else
  {
    v19 = a5;
    v22 = 773;
    v20 = a6;
    v21 = " component cannot be empty";
    v8 = sub_C63BB0();
    v10 = v9;
    v11 = v8;
  }
  sub_CA0F50(v17, &v19);
  sub_C63F00(a1, v17, v11, v10);
  if ( (__int64 *)v17[0] != &v18 )
    j_j___libc_free_0(v17[0], v18 + 1);
  return a1;
}
