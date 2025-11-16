// Function: sub_2FB1CD0
// Address: 0x2fb1cd0
//
__int64 __fastcall sub_2FB1CD0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r15d
  unsigned int v8; // eax
  __int64 v9; // rdx
  int v10; // ecx
  __int64 v11; // r13
  __int64 v12; // r14
  unsigned __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 *v15; // rax
  unsigned int v16; // r8d
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 *v20; // r14
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r8
  _QWORD *v24; // rax
  _QWORD *v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD *)(a1[5] + 112LL);
  v8 = v7 & 0x7FFFFFFF;
  v9 = v7 & 0x7FFFFFFF;
  v10 = *(_DWORD *)(*(_QWORD *)(a1[1] + 80LL) + 4 * v9);
  if ( v10 )
  {
    v7 = *(_DWORD *)(*(_QWORD *)(a1[1] + 80LL) + 4 * v9);
    v8 = v10 & 0x7FFFFFFF;
    v9 = v10 & 0x7FFFFFFF;
  }
  v11 = a1[2];
  v12 = 8 * v9;
  v13 = *(unsigned int *)(v11 + 160);
  if ( (unsigned int)v13 <= v8 || (v14 = *(_QWORD *)(*(_QWORD *)(v11 + 152) + 8 * v9)) == 0 )
  {
    v18 = v8 + 1;
    if ( (unsigned int)v13 < v18 && v18 != v13 )
    {
      if ( v18 >= v13 )
      {
        v22 = *(_QWORD *)(v11 + 168);
        v23 = v18 - v13;
        if ( v18 > (unsigned __int64)*(unsigned int *)(v11 + 164) )
        {
          v26 = v18 - v13;
          sub_C8D5F0(v11 + 152, (const void *)(v11 + 168), v18, 8u, v23, a6);
          v13 = *(unsigned int *)(v11 + 160);
          v23 = v26;
        }
        v19 = *(_QWORD *)(v11 + 152);
        v24 = (_QWORD *)(v19 + 8 * v13);
        v25 = &v24[v23];
        if ( v24 != v25 )
        {
          do
            *v24++ = v22;
          while ( v25 != v24 );
          LODWORD(v13) = *(_DWORD *)(v11 + 160);
          v19 = *(_QWORD *)(v11 + 152);
        }
        *(_DWORD *)(v11 + 160) = v23 + v13;
        goto LABEL_13;
      }
      *(_DWORD *)(v11 + 160) = v18;
    }
    v19 = *(_QWORD *)(v11 + 152);
LABEL_13:
    v20 = (__int64 *)(v19 + v12);
    v21 = sub_2E10F30(v7);
    *v20 = v21;
    v14 = v21;
    sub_2E11E80((_QWORD *)v11, v21);
  }
  v15 = (__int64 *)sub_2E09D00((__int64 *)v14, a2);
  if ( v15 == (__int64 *)(*(_QWORD *)v14 + 24LL * *(unsigned int *)(v14 + 8))
    || (v16 = a2 & 0xFFFFFFF8,
        (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(a2 >> 1) & 3)) )
  {
    v16 = 0;
    if ( *(__int64 **)v14 != v15 )
      LOBYTE(v16) = *(v15 - 2) == a2;
  }
  else
  {
    LOBYTE(v16) = *v15 == a2;
  }
  return v16;
}
