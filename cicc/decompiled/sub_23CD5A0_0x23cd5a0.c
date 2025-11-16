// Function: sub_23CD5A0
// Address: 0x23cd5a0
//
__int64 __fastcall sub_23CD5A0(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // r15
  _QWORD *v8; // r13
  _QWORD *i; // rbx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 result; // rax

  v3 = a2 / 0xC + 3;
  v4 = a2 / 0xC + 1;
  if ( v3 > 8 )
  {
    a1[1] = v3;
    if ( v3 > 0xFFFFFFFFFFFFFFFLL )
      sub_4261EA(v3, a2, a2 / 0xC);
    v5 = 8 * v3;
  }
  else
  {
    a1[1] = 8;
    v5 = 64;
  }
  v6 = sub_22077B0(v5);
  *a1 = v6;
  v7 = (_QWORD *)(v6 + 8 * ((a1[1] - v4) >> 1));
  v8 = &v7[v4];
  for ( i = v7; v8 > i; *(i - 1) = sub_22077B0(0x1E0u) )
    ++i;
  v10 = *v7;
  v11 = *(v8 - 1);
  a1[5] = (__int64)v7;
  a1[3] = v10;
  a1[4] = v10 + 480;
  a1[9] = (__int64)(v8 - 1);
  a1[8] = v11 + 480;
  a1[2] = v10;
  a1[7] = v11;
  result = v11 + 40 * (a2 % 0xC);
  a1[6] = result;
  return result;
}
