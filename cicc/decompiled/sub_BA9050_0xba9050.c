// Function: sub_BA9050
// Address: 0xba9050
//
__int64 __fastcall sub_BA9050(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 v7; // rsi
  int v8; // eax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // r13
  unsigned __int64 *v12; // rcx
  unsigned __int64 v13; // rdx

  v3 = sub_B91B20((__int64)a2);
  v5 = v4;
  v6 = sub_C92610(v3, v4);
  v7 = v3;
  v8 = sub_C92860(a1 + 288, v3, v5, v6);
  if ( v8 != -1 )
  {
    v9 = *(_QWORD *)(a1 + 288);
    v10 = (_QWORD *)(v9 + 8LL * v8);
    if ( v10 != (_QWORD *)(v9 + 8LL * *(unsigned int *)(a1 + 296)) )
    {
      v11 = (_QWORD *)*v10;
      sub_C929B0(a1 + 288, *v10);
      v7 = *v11 + 17LL;
      sub_C7D6A0(v11, v7, 8);
    }
  }
  if ( *(_QWORD **)(a1 + 864) == a2 )
    *(_QWORD *)(a1 + 864) = 0;
  v12 = (unsigned __int64 *)a2[1];
  v13 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v12 = v13 | *v12 & 7;
  *(_QWORD *)(v13 + 8) = v12;
  *a2 &= 7uLL;
  a2[1] = 0;
  sub_B91A80(a2, v7);
  return j_j___libc_free_0(a2, 64);
}
