// Function: sub_EE6060
// Address: 0xee6060
//
void *__fastcall sub_EE6060(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // r15
  __int64 v3; // rcx
  _BYTE *v4; // r13
  size_t v5; // rdx
  unsigned __int64 v6; // rax
  void *v7; // r8
  __int64 v9; // rax

  v2 = (_BYTE *)a1[3];
  v3 = a1[101];
  v4 = (_BYTE *)(8 * a2 + a1[2]);
  v5 = v2 - v4;
  a1[111] += v2 - v4;
  v6 = (v3 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[102] >= v2 - v4 + v6 && v3 )
  {
    a1[101] = v5 + v6;
    v7 = (void *)((v3 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    v9 = sub_9D1E70((__int64)(a1 + 101), v5, v5, 3);
    v5 = v2 - v4;
    v7 = (void *)v9;
  }
  if ( v4 != v2 )
    v7 = memmove(v7, v4, v5);
  a1[3] = a1[2] + 8 * a2;
  return v7;
}
