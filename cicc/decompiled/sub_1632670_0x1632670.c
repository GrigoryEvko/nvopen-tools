// Function: sub_1632670
// Address: 0x1632670
//
__int64 __fastcall sub_1632670(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // eax
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // r14
  unsigned __int64 *v8; // rcx
  unsigned __int64 v9; // rdx

  v2 = *(_QWORD *)(a1 + 272);
  v3 = sub_161F640((__int64)a2);
  v5 = sub_16D1B30(v2, v3, v4);
  if ( v5 != -1 )
  {
    v6 = (unsigned __int64 *)(*(_QWORD *)v2 + 8LL * v5);
    if ( v6 != (unsigned __int64 *)(*(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8)) )
    {
      v7 = *v6;
      sub_16D1CB0(v2, *v6);
      _libc_free(v7);
    }
  }
  v8 = (unsigned __int64 *)a2[1];
  v9 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v8 = v9 | *v8 & 7;
  *(_QWORD *)(v9 + 8) = v8;
  *a2 &= 7uLL;
  a2[1] = 0;
  sub_161F5A0(a2);
  return j_j___libc_free_0(a2, 64);
}
