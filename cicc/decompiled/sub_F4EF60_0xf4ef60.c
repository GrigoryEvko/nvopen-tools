// Function: sub_F4EF60
// Address: 0xf4ef60
//
__int64 __fastcall sub_F4EF60(_QWORD *a1)
{
  _QWORD *v1; // rsi
  __int64 v2; // rax
  __int64 v3; // rdx
  bool v4; // zf
  __int64 *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // rdi

  v1 = (_QWORD *)a1[2];
  v2 = 3;
  v3 = (__int64)(a1[3] - (_QWORD)v1) >> 3;
  v4 = *v1 == 4101;
  if ( *v1 != 4101 )
    v2 = 1;
  v5 = &v1[v2];
  v6 = a1[1];
  v7 = v3 - (2LL * v4 + 1);
  v8 = (__int64 *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v6 & 4) != 0 )
    v8 = (__int64 *)*v8;
  return sub_B0D000(v8, v5, v7, 0, 1);
}
