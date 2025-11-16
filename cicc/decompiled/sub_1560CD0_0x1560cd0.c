// Function: sub_1560CD0
// Address: 0x1560cd0
//
__int64 __fastcall sub_1560CD0(__int64 *a1, int a2, _QWORD *a3)
{
  __int64 result; // rax
  unsigned __int64 v5; // rdx
  int v6; // r8d
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 *v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // [rsp+8h] [rbp-88h]
  __int64 *v12; // [rsp+10h] [rbp-80h] BYREF
  __int64 v13; // [rsp+18h] [rbp-78h]
  _BYTE v14[8]; // [rsp+20h] [rbp-70h] BYREF
  char v15; // [rsp+28h] [rbp-68h] BYREF

  if ( !sub_1560CB0(a3) )
    return 0;
  if ( a2 == -1 )
  {
    v10 = (__int64 *)&v15;
    v7 = 0;
    v13 = 0x800000001LL;
    v9 = (__int64 *)v14;
    v12 = (__int64 *)v14;
  }
  else
  {
    v5 = (unsigned int)(a2 + 2);
    v13 = 0x800000000LL;
    v6 = a2 + 2;
    v12 = (__int64 *)v14;
    v7 = (unsigned int)(a2 + 1);
    v8 = v5;
    v9 = (__int64 *)v14;
    if ( v5 > 8 )
    {
      sub_16CD150(&v12, v14, v5, 8);
      v9 = v12;
      v6 = a2 + 2;
    }
    v10 = &v9[v8];
    LODWORD(v13) = v6;
    if ( v9 == v10 )
      goto LABEL_12;
  }
  do
  {
    if ( v9 )
      *v9 = 0;
    ++v9;
  }
  while ( v9 != v10 );
  v10 = v12;
LABEL_12:
  v10[v7] = sub_1560BF0(a1, a3);
  result = sub_155F990(a1, v12, (unsigned int)v13);
  if ( v12 != (__int64 *)v14 )
  {
    v11 = result;
    _libc_free((unsigned __int64)v12);
    return v11;
  }
  return result;
}
