// Function: sub_D53100
// Address: 0xd53100
//
__int64 __fastcall sub_D53100(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // r15
  _QWORD *v9; // r13
  _QWORD *i; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 result; // rax

  v4 = (a2 >> 4) + 1;
  v5 = (a2 >> 4) + 3;
  if ( v5 > 8 )
  {
    a1[1] = v5;
    if ( v5 > 0xFFFFFFFFFFFFFFFLL )
      sub_4261EA(v5, a2, a3);
    v6 = 8 * v5;
  }
  else
  {
    a1[1] = 8;
    v6 = 64;
  }
  v7 = sub_22077B0(v6);
  *a1 = v7;
  v8 = (_QWORD *)(v7 + 8 * ((unsigned __int64)(a1[1] - v4) >> 1));
  v9 = &v8[v4];
  for ( i = v8; v9 > i; *(i - 1) = sub_22077B0(512) )
    ++i;
  v11 = *v8;
  a1[5] = (__int64)v8;
  a1[3] = v11;
  a1[4] = v11 + 512;
  a1[9] = (__int64)(v9 - 1);
  v12 = *(v9 - 1);
  a1[2] = v11;
  a1[7] = v12;
  v13 = v12 + 512;
  result = 32 * (a2 & 0xF) + v12;
  a1[8] = v13;
  a1[6] = result;
  return result;
}
