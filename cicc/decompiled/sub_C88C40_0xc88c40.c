// Function: sub_C88C40
// Address: 0xc88c40
//
__int64 __fastcall sub_C88C40(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rcx
  __int64 v5; // rdx
  __int64 v6; // r8
  int *v7; // r15
  int *v8; // r14
  __int64 v9; // rax
  int *v10; // rbx
  _BYTE *v11; // rdx
  _BYTE *v12; // rsi
  int *v13; // r8
  int v14; // eax
  int *v16; // [rsp+0h] [rbp-70h]
  __int64 v17; // [rsp+0h] [rbp-70h]
  __int64 v18; // [rsp+0h] [rbp-70h]
  size_t n; // [rsp+8h] [rbp-68h]
  int v20; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v22; // [rsp+28h] [rbp-48h]
  _BYTE *v23; // [rsp+30h] [rbp-40h]

  v3 = 5489;
  v5 = 1;
  *a1 = 5489;
  while ( 1 )
  {
    a1[v5] = v5 + 0x5851F42D4C957F2DLL * (v3 ^ (v3 >> 62));
    if ( ++v5 == 312 )
      break;
    v3 = a1[v5 - 1];
  }
  a1[312] = 312;
  n = a3 + 2;
  if ( a3 == -2 )
  {
    v6 = 8;
    v7 = 0;
    v8 = 0;
  }
  else
  {
    if ( n > 0x1FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_default_append");
    n *= 4LL;
    v8 = (int *)sub_22077B0(n);
    v7 = (int *)((char *)v8 + n);
    memset(v8, 0, n);
    v6 = (__int64)(v8 + 2);
  }
  if ( !qword_4F84130 )
  {
    v17 = v6;
    sub_C7D570(&qword_4F84130, sub_C883E0, (__int64)sub_C883C0);
    v6 = v17;
  }
  *v8 = *(_QWORD *)(qword_4F84130 + 136);
  if ( !qword_4F84130 )
  {
    v18 = v6;
    sub_C7D570(&qword_4F84130, sub_C883E0, (__int64)sub_C883C0);
    v6 = v18;
  }
  v8[1] = *(_DWORD *)(qword_4F84130 + 140);
  v9 = 0;
  if ( a3 > 0 )
  {
    do
    {
      *(_DWORD *)(v6 + 4 * v9) = *(char *)(a2 + v9);
      ++v9;
    }
    while ( a3 != v9 );
  }
  v21 = 0;
  v22 = 0;
  v23 = 0;
  if ( v7 != v8 )
  {
    v10 = v8;
    v11 = 0;
    v12 = 0;
    v13 = &v20;
    while ( 1 )
    {
      v14 = *v10;
      v20 = *v10;
      if ( v12 == v11 )
      {
        ++v10;
        v16 = v13;
        sub_C88AB0((__int64)&v21, v12, v13);
        v13 = v16;
        if ( v7 == v10 )
          break;
      }
      else
      {
        if ( v12 )
        {
          *(_DWORD *)v12 = v14;
          v12 = v22;
        }
        ++v10;
        v22 = v12 + 4;
        if ( v7 == v10 )
          break;
      }
      v12 = v22;
      v11 = v23;
    }
  }
  sub_C88640(a1, &v21);
  if ( v21 )
    j_j___libc_free_0(v21, &v23[-v21]);
  return j_j___libc_free_0(v8, n);
}
