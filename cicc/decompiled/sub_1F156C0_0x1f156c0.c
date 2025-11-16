// Function: sub_1F156C0
// Address: 0x1f156c0
//
void __fastcall sub_1F156C0(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v8; // r15
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 *v12; // rcx
  __int64 *v13; // rax
  __int64 v14; // r13
  unsigned __int64 v15; // rsi
  __int64 i; // [rsp+10h] [rbp-40h]
  __int64 *v19; // [rsp+18h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 64);
  for ( i = *(_QWORD *)(a2 + 72); i != v8; v8 += 8 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL) + 392LL)
                    + 16LL * *(unsigned int *)(*(_QWORD *)v8 + 48LL)
                    + 8);
    v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
    if ( ((v14 >> 1) & 3) != 0 )
      v10 = (2LL * (int)(((v14 >> 1) & 3) - 1)) | v15;
    else
      v10 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
    v12 = (__int64 *)v11;
    if ( a5 != -1 )
    {
      do
        v11 = *(_QWORD *)(v11 + 104);
      while ( *(_DWORD *)(v11 + 112) != a5 );
      v12 = (__int64 *)v11;
    }
    v19 = v12;
    v13 = (__int64 *)sub_1DB3C70(v12, v10);
    if ( v13 != (__int64 *)(*v19 + 24LL * *((unsigned int *)v19 + 2))
      && (*(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v13 >> 1) & 3) <= (*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v10 >> 1)
                                                                                             & 3) )
    {
      sub_1DC5C40(a3, a4, v14, 0, a7, a8);
    }
  }
}
