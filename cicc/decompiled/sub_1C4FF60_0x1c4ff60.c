// Function: sub_1C4FF60
// Address: 0x1c4ff60
//
__int64 __fastcall sub_1C4FF60(
        _DWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // r12d
  _QWORD *v14; // rdi
  __int64 v15; // rbx
  char *v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // r14d
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rsi
  int v23; // eax
  unsigned int v24; // eax
  void *s1; // [rsp+10h] [rbp-50h] BYREF
  size_t n; // [rsp+18h] [rbp-48h]
  _QWORD v27[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( dword_4FBB960 > 0 )
  {
    *a1 = 1;
    if ( dword_4FBB960 <= 0 )
      return 0;
  }
  if ( dword_4FBB880 > 0 )
    a1[1] = 0;
  v11 = qword_4FBBA40;
  v12 = 0;
  v13 = 0;
  if ( qword_4FBBA40 == qword_4FBBA48 )
  {
LABEL_19:
    v20 = *(_QWORD *)(a2 + 80);
    v21 = a2 + 72;
    v18 = 0;
    if ( v20 == a2 + 72 )
      return 0;
    do
    {
      v22 = v20 - 24;
      if ( !v20 )
        v22 = 0;
      if ( dword_4FBB880 <= 0 || (v23 = a1[1] + 1, a1[1] = v23, v23 <= dword_4FBB880) )
      {
        v24 = sub_1C4D210((__int64)a1, v22, a3, a4, a5, a6, a7, a8, a9, a10);
        if ( (_BYTE)v24 )
          v18 = v24;
      }
      v20 = *(_QWORD *)(v20 + 8);
    }
    while ( v21 != v20 );
    return v18;
  }
  while ( 1 )
  {
    v15 = v11 + 32 * v12;
    v16 = (char *)sub_1649960(a2);
    s1 = v27;
    if ( v16 )
      break;
    n = 0;
    LOBYTE(v27[0]) = 0;
    if ( !*(_QWORD *)(v15 + 8) )
      return 0;
LABEL_10:
    v11 = qword_4FBBA40;
    v12 = (unsigned int)++v13;
    if ( v13 == (qword_4FBBA48 - qword_4FBBA40) >> 5 )
      goto LABEL_19;
  }
  sub_1C4A290((__int64 *)&s1, v16, (__int64)&v16[v17]);
  v14 = s1;
  if ( *(_QWORD *)(v15 + 8) != n || n && (v14 = s1, memcmp(s1, *(const void **)v15, n)) )
  {
    if ( v14 != v27 )
      j_j___libc_free_0(v14, v27[0] + 1LL);
    goto LABEL_10;
  }
  if ( v14 != v27 )
    j_j___libc_free_0(v14, v27[0] + 1LL);
  return 0;
}
