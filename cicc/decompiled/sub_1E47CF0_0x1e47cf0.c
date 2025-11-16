// Function: sub_1E47CF0
// Address: 0x1e47cf0
//
__int64 __fastcall sub_1E47CF0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // r15
  _QWORD *v8; // r13
  _QWORD *i; // rbx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 result; // rax

  v3 = (a2 >> 6) + 1;
  v4 = (a2 >> 6) + 3;
  if ( v4 > 8 )
  {
    a1[1] = v4;
    v5 = 8 * v4;
  }
  else
  {
    a1[1] = 8;
    v5 = 64;
  }
  v6 = sub_22077B0(v5);
  *a1 = v6;
  v7 = (_QWORD *)(v6 + 8 * ((unsigned __int64)(a1[1] - v3) >> 1));
  v8 = &v7[v3];
  for ( i = v7; v8 > i; *(i - 1) = sub_22077B0(512) )
    ++i;
  v10 = *v7;
  a1[5] = (__int64)v7;
  a1[3] = v10;
  a1[4] = v10 + 512;
  a1[9] = (__int64)(v8 - 1);
  v11 = *(v8 - 1);
  a1[2] = v10;
  a1[7] = v11;
  v12 = v11 + 512;
  result = v11 + 8 * (a2 & 0x3F);
  a1[8] = v12;
  a1[6] = result;
  return result;
}
