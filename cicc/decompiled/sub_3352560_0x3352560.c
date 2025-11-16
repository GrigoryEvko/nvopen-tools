// Function: sub_3352560
// Address: 0x3352560
//
__int64 __fastcall sub_3352560(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  _QWORD *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r8
  unsigned int *v7; // r10
  __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rbx
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // rdx
  unsigned __int64 *v18; // rdi
  unsigned int v19; // r12d
  __int64 v21; // rsi
  _QWORD *v22; // rcx
  int v23; // edi
  unsigned int v24; // esi
  unsigned int v25; // eax
  int v26; // esi
  unsigned __int64 v27; // [rsp+0h] [rbp-150h]
  _QWORD *v29; // [rsp+10h] [rbp-140h] BYREF
  __int64 v30; // [rsp+18h] [rbp-138h]
  _QWORD v31[38]; // [rsp+20h] [rbp-130h] BYREF

  v29 = v31;
  v31[1] = 0;
  v31[0] = a1;
  v4 = v31;
  v30 = 0x1000000001LL;
  v5 = 1;
  do
  {
    v6 = *a2;
    v7 = (unsigned int *)&v4[2 * v5 - 2];
    v8 = v7[2];
    v9 = *(unsigned int *)(*(_QWORD *)v7 + 48LL);
    v10 = *(_QWORD **)(*(_QWORD *)v7 + 40LL);
    if ( (unsigned int)v8 < (unsigned int)v9 )
    {
      v11 = (unsigned int)(v8 + 1);
      v12 = v11 + (unsigned int)(v9 - v8 - 1) + 1;
      while ( 1 )
      {
        v13 = v10[2 * v8];
        if ( (v13 & 6) == 0 )
        {
          v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !*(_DWORD *)(v6 + 4LL * *(unsigned int *)(v14 + 200)) )
          {
            v7[2] = v11;
            v15 = (unsigned int)v30;
            v16 = v2 & 0xFFFFFFFF00000000LL;
            v17 = (unsigned int)v30 + 1LL;
            v2 &= 0xFFFFFFFF00000000LL;
            if ( v17 > HIDWORD(v30) )
            {
              v27 = v16;
              sub_C8D5F0((__int64)&v29, v31, v17, 0x10u, v16, v12);
              v4 = v29;
              v15 = (unsigned int)v30;
              v16 = v27;
            }
            v18 = &v4[2 * v15];
            *v18 = v14;
            v18[1] = v16;
            v5 = (unsigned int)(v30 + 1);
            LODWORD(v30) = v30 + 1;
            goto LABEL_11;
          }
        }
        v8 = v11;
        if ( v12 == v11 + 1 )
          break;
        ++v11;
      }
    }
    v21 = 2 * v9;
    v22 = &v10[v21];
    if ( &v10[v21] == v10 )
    {
      v26 = 1;
      goto LABEL_24;
    }
    v23 = 0;
    v24 = 0;
    do
    {
      while ( 1 )
      {
        if ( (*v10 & 6) != 0 )
          goto LABEL_18;
        v25 = *(_DWORD *)(v6 + 4LL * *(unsigned int *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 200));
        if ( v25 > v24 )
          break;
        v23 += v25 == v24;
LABEL_18:
        v10 += 2;
        if ( v22 == v10 )
          goto LABEL_22;
      }
      v10 += 2;
      v24 = v25;
      v23 = 0;
    }
    while ( v22 != v10 );
LABEL_22:
    v26 = v23 + v24;
    if ( !v26 )
      v26 = 1;
LABEL_24:
    *(_DWORD *)(v6 + 4LL * *(unsigned int *)(*(_QWORD *)v7 + 200LL)) = v26;
    v5 = (unsigned int)(v30 - 1);
    LODWORD(v30) = v30 - 1;
LABEL_11:
    v4 = v29;
  }
  while ( (_DWORD)v5 );
  v19 = *(_DWORD *)(*a2 + 4LL * *(unsigned int *)(a1 + 200));
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
  return v19;
}
