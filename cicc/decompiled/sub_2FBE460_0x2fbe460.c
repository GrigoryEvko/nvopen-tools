// Function: sub_2FBE460
// Address: 0x2fbe460
//
__int64 __fastcall sub_2FBE460(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 *v6; // rdx
  int *v8; // r15
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // r10
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // r14
  __int64 *v17; // rax
  unsigned __int64 v18; // r13
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // r11
  __int64 v22; // r14
  unsigned __int64 v23; // rax
  _QWORD *v24; // rcx
  _QWORD *v25; // rdi
  __int64 v26; // [rsp+0h] [rbp-50h]
  unsigned __int64 v27; // [rsp+8h] [rbp-48h]
  _QWORD *v28; // [rsp+10h] [rbp-40h]
  int v29; // [rsp+10h] [rbp-40h]
  __int64 *v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 152LL) + 16LL * *(unsigned int *)(a2 + 24));
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v6 = (__int64 *)sub_2E09D00((__int64 *)v5, v4);
  if ( v6 == (__int64 *)(*(_QWORD *)v5 + 24LL * *(unsigned int *)(v5 + 8)) )
    return v4;
  if ( (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v6 >> 1) & 3) > (*(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                        | (unsigned int)(v4 >> 1) & 3) )
    return v4;
  v8 = (int *)v6[2];
  if ( !v8 )
    return v4;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(unsigned int *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                        + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL));
  v11 = *(unsigned int *)(v9 + 160);
  v12 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL) + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL))
      & 0x7FFFFFFF;
  v13 = 8LL * v12;
  if ( v12 >= (unsigned int)v11 || (v16 = *(_QWORD *)(*(_QWORD *)(v9 + 152) + 8LL * v12)) == 0 )
  {
    v14 = v12 + 1;
    if ( (unsigned int)v11 < v14 )
    {
      v21 = v14;
      if ( v14 != v11 )
      {
        if ( v14 >= v11 )
        {
          v22 = *(_QWORD *)(v9 + 168);
          v23 = v14 - v11;
          if ( v21 > *(unsigned int *)(v9 + 164) )
          {
            v26 = v13;
            v27 = v23;
            v29 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                            + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL));
            v31 = *(_QWORD *)(a1 + 8);
            sub_C8D5F0(v9 + 152, (const void *)(v9 + 168), v21, 8u, v9, v10);
            v9 = v31;
            v13 = v26;
            v23 = v27;
            LODWORD(v10) = v29;
            v11 = *(unsigned int *)(v31 + 160);
          }
          v15 = *(_QWORD *)(v9 + 152);
          v24 = (_QWORD *)(v15 + 8 * v11);
          v25 = &v24[v23];
          if ( v24 != v25 )
          {
            do
              *v24++ = v22;
            while ( v25 != v24 );
            LODWORD(v11) = *(_DWORD *)(v9 + 160);
            v15 = *(_QWORD *)(v9 + 152);
          }
          *(_DWORD *)(v9 + 160) = v23 + v11;
          goto LABEL_8;
        }
        *(_DWORD *)(v9 + 160) = v14;
      }
    }
    v15 = *(_QWORD *)(v9 + 152);
LABEL_8:
    v28 = (_QWORD *)v9;
    v30 = (__int64 *)(v15 + v13);
    v16 = sub_2E10F30(v10);
    *v30 = v16;
    sub_2E11E80(v28, v16);
  }
  v17 = (__int64 *)sub_2E312E0(a2, *(_QWORD *)(a2 + 56), *(unsigned int *)(v16 + 112), 1);
  v18 = sub_2FB9FE0((__int64 *)a1, 0, v8, v4, a2, v17);
  sub_2FBD6E0(a1 + 192, v4, *(_QWORD *)(v18 + 8), *(unsigned int *)(a1 + 80), v19, v20);
  return *(_QWORD *)(v18 + 8);
}
