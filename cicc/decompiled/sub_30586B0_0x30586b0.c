// Function: sub_30586B0
// Address: 0x30586b0
//
unsigned __int64 __fastcall sub_30586B0(int a1, __int64 *a2, _QWORD *a3)
{
  _QWORD *v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  _QWORD *v7; // rdi
  _QWORD *i; // rbx
  _QWORD *v10; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v11; // [rsp+8h] [rbp-48h]

  v4 = sub_C33340();
  if ( (_QWORD *)*a2 == v4 )
    sub_C3C790(&v10, (_QWORD **)a2);
  else
    sub_C33EB0(&v10, a2);
  v5 = a3[24];
  a3[34] += 56LL;
  v6 = (v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3[25] >= v6 + 56 && v5 )
    a3[24] = v6 + 56;
  else
    v6 = sub_9D1E70((__int64)(a3 + 24), 56, 56, 3);
  *(_DWORD *)(v6 + 24) = a1;
  v7 = (_QWORD *)(v6 + 32);
  *(_DWORD *)(v6 + 8) = 4;
  *(_QWORD *)(v6 + 16) = 0;
  *(_QWORD *)v6 = &unk_4A2F1C0;
  if ( v10 == v4 )
    sub_C3C840(v7, &v10);
  else
    sub_C338E0((__int64)v7, (__int64)&v10);
  if ( v10 == v4 )
  {
    if ( v11 )
    {
      for ( i = &v11[3 * *(v11 - 1)]; v11 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v10);
  }
  return v6;
}
