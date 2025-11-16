// Function: sub_20FF4D0
// Address: 0x20ff4d0
//
void __fastcall sub_20FF4D0(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _DWORD *v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 v9; // rdx
  __int64 *v10; // r15
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 *v13; // rdi
  unsigned int v14; // r9d
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // edi
  unsigned __int64 v19; // r9
  int i; // edx
  __int64 v21; // r10
  int v22; // eax
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // rsi
  unsigned __int64 v26; // rax
  int v27; // edx
  __int64 v28; // rsi
  _DWORD *v29; // [rsp-98h] [rbp-98h] BYREF
  _QWORD *v30; // [rsp-90h] [rbp-90h] BYREF
  __int64 v31; // [rsp-88h] [rbp-88h]
  _QWORD v32[16]; // [rsp-80h] [rbp-80h] BYREF

  if ( !*(_DWORD *)(a3 + 8) )
    return;
  v6 = a1 + 2;
  ++*a1;
  v7 = *(unsigned int *)(a3 + 8);
  v8 = *(__int64 **)a3;
  v9 = (unsigned int)a1[50];
  v10 = &v8[3 * v7];
  v11 = *v8;
  v29 = a1 + 2;
  v30 = v32;
  v31 = 0x400000000LL;
  if ( !(_DWORD)v9 )
  {
    v12 = (unsigned int)a1[51];
    if ( (_DWORD)v12 )
    {
      v13 = (__int64 *)(a1 + 4);
      v14 = *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v11 >> 1) & 3;
      do
      {
        if ( (*(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v13 >> 1) & 3) > v14 )
          break;
        v9 = (unsigned int)(v9 + 1);
        v13 += 2;
      }
      while ( (_DWORD)v12 != (_DWORD)v9 );
    }
    v32[0] = v6;
    LODWORD(v31) = 1;
    v32[1] = v12 | (v9 << 32);
LABEL_8:
    v15 = (unsigned __int64)v30;
    while ( *(_DWORD *)(v15 + 12) < *(_DWORD *)(v15 + 8) )
    {
      v16 = v8[1];
      v17 = *v8;
      v8 += 3;
      sub_20FF400((__int64)&v29, v17, v16, a2);
      if ( v8 == v10 )
        goto LABEL_26;
      if ( !(_DWORD)v31 )
        break;
      v15 = (unsigned __int64)v30;
      if ( *((_DWORD *)v30 + 3) < *((_DWORD *)v30 + 2) )
      {
        if ( v29[48] )
        {
          sub_20F82D0((__int64)&v29, *v8);
          v22 = v31;
        }
        else
        {
          v18 = v29[49];
          v19 = (unsigned __int64)&v30[2 * (unsigned int)v31 - 2];
          for ( i = *(_DWORD *)(v19 + 12); v18 != i; ++i )
          {
            v21 = *(_QWORD *)&v29[4 * i + 2];
            if ( (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v21 >> 1) & 3) > (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(*v8 >> 1) & 3) )
              break;
          }
          *(_DWORD *)(v19 + 12) = i;
          v22 = v31;
        }
        if ( v22 )
          goto LABEL_8;
        goto LABEL_20;
      }
    }
    goto LABEL_20;
  }
  sub_20FCFA0((__int64)&v29, v11, v9, a4, a5, v11);
  if ( (_DWORD)v31 )
    goto LABEL_8;
LABEL_20:
  v23 = v10 - 3;
  sub_20FF400((__int64)&v29, *(v10 - 3), *(v10 - 2), a2);
  if ( v10 - 3 != v8 )
  {
    do
    {
      while ( 1 )
      {
        v24 = v8[1];
        v25 = *v8;
        v8 += 3;
        sub_20FF400((__int64)&v29, v25, v24, a2);
        v26 = (unsigned __int64)&v30[2 * (unsigned int)v31 - 2];
        v27 = *(_DWORD *)(v26 + 12) + 1;
        *(_DWORD *)(v26 + 12) = v27;
        if ( v27 == LODWORD(v30[2 * (unsigned int)v31 - 1]) )
        {
          v28 = (unsigned int)v29[48];
          if ( (_DWORD)v28 )
            break;
        }
        if ( v23 == v8 )
          goto LABEL_26;
      }
      sub_39460A0(&v30, v28);
    }
    while ( v23 != v8 );
  }
LABEL_26:
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
}
