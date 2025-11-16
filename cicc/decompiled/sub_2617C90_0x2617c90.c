// Function: sub_2617C90
// Address: 0x2617c90
//
__int64 __fastcall sub_2617C90(_DWORD *a1, const void *a2, __int64 a3, __int64 a4, __int64 a5)
{
  const void *v5; // r9
  size_t v6; // r14
  unsigned __int64 v7; // r12
  __int64 *v9; // rdi
  int v10; // eax
  __int64 *v11; // r10
  __int64 *v12; // r13
  __int64 *v13; // r14
  unsigned int v14; // r12d
  __int64 v15; // r15
  int v18; // [rsp+10h] [rbp-90h]
  __int64 *v20; // [rsp+20h] [rbp-80h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  _BYTE v22[112]; // [rsp+30h] [rbp-70h] BYREF

  v5 = (const void *)a3;
  v6 = a3 - (_QWORD)a2;
  v7 = (a3 - (__int64)a2) >> 3;
  v20 = (__int64 *)v22;
  v18 = a5;
  v21 = 0x800000000LL;
  if ( (unsigned __int64)(a3 - (_QWORD)a2) > 0x40 )
  {
    sub_C8D5F0((__int64)&v20, v22, v7, 8u, a5, a3);
    v9 = v20;
    v10 = v21;
    v5 = (const void *)a3;
    v11 = &v20[(unsigned int)v21];
  }
  else
  {
    v9 = (__int64 *)v22;
    v10 = 0;
    v11 = (__int64 *)v22;
  }
  if ( v5 != a2 )
  {
    memmove(v11, a2, v6);
    v9 = v20;
    v10 = v21;
  }
  LODWORD(v21) = v7 + v10;
  v12 = &v9[(unsigned int)(v7 + v10)];
  if ( v12 == v9 )
  {
    v14 = 0;
  }
  else
  {
    v13 = v9;
    v14 = 0;
    do
    {
      v15 = *v13;
      if ( (unsigned __int8)sub_D4B3D0(*v13) )
      {
        v14 |= sub_2617A30((__int64)a1, v15, a4, v18);
        if ( !*a1 )
          break;
      }
      ++v13;
    }
    while ( v12 != v13 );
    v9 = v20;
  }
  if ( v9 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v9);
  return v14;
}
