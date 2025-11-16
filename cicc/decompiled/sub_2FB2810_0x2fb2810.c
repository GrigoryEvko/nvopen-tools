// Function: sub_2FB2810
// Address: 0x2fb2810
//
void __fastcall sub_2FB2810(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v8; // rbx
  __int64 *v9; // r13
  signed __int64 v10; // r12
  __int64 *v11; // rax
  __int64 v12; // r15
  __int64 i; // [rsp+20h] [rbp-40h]
  __int64 v18; // [rsp+28h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 64);
  v18 = v8 + 8LL * *(unsigned int *)(a2 + 72);
  for ( i = a5 & a6; v18 != v8; v8 += 8 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 152LL)
                    + 16LL * *(unsigned int *)(*(_QWORD *)v8 + 24LL)
                    + 8);
    if ( ((v12 >> 1) & 3) != 0 )
    {
      v10 = v12 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v12 >> 1) & 3) - 1));
      v9 = *(__int64 **)(*(_QWORD *)(a1 + 72) + 8LL);
      if ( i != -1 )
LABEL_10:
        v9 = sub_2FB2180(a5, a6, (__int64)v9);
    }
    else
    {
      v9 = *(__int64 **)(*(_QWORD *)(a1 + 72) + 8LL);
      v10 = *(_QWORD *)(v12 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
      if ( i != -1 )
        goto LABEL_10;
    }
    v11 = (__int64 *)sub_2E09D00(v9, v10);
    if ( v11 != (__int64 *)(*v9 + 24LL * *((unsigned int *)v9 + 2))
      && (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3) <= (*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v10 >> 1)
                                                                                             & 3) )
    {
      sub_2E20270(a3, a4, v12, 0, a7, a8);
    }
  }
}
