// Function: sub_34B51C0
// Address: 0x34b51c0
//
__int64 __fastcall sub_34B51C0(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // r15d
  unsigned int v10; // r12d
  char v11; // r15
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r12
  _QWORD *v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rax
  _QWORD *v23; // rdx
  __int64 v24; // rcx
  unsigned int v26; // edi
  unsigned int v27; // edx
  __int64 v28; // rax
  _QWORD *v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rcx
  int v32; // ecx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // r10
  int v35; // eax
  __int64 v36; // r15
  int v37; // eax
  void *s; // [rsp+8h] [rbp-98h]
  __int64 v39; // [rsp+10h] [rbp-90h]
  unsigned int v40; // [rsp+1Ch] [rbp-84h] BYREF
  char *v41; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-78h]
  char v43; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v44; // [rsp+60h] [rbp-40h]

  v8 = a2[4];
  v40 = a3;
  v9 = *(_DWORD *)(v8 + 16);
  s = (void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  v10 = (unsigned int)(v9 + 63) >> 6;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  if ( v10 > 6 )
  {
    sub_C8D5F0(a1, s, v10, 8u, a5, a6);
    memset(*(void **)a1, 0, 8LL * v10);
    *(_DWORD *)(a1 + 8) = v10;
  }
  else
  {
    if ( v10 )
      memset((void *)(a1 + 16), 0, 8LL * v10);
    *(_DWORD *)(a1 + 8) = v10;
  }
  *(_DWORD *)(a1 + 64) = v9;
  v11 = 1;
  v12 = sub_34B4480(a2[15] + 56, &v40);
  v39 = v15;
  v16 = v12;
  if ( v15 != v12 )
  {
    while ( 1 )
    {
      v17 = *(_QWORD **)(v16 + 48);
      if ( v17 )
        break;
LABEL_15:
      v16 = sub_220EEE0(v16);
      if ( v39 == v16 )
        return a1;
    }
    sub_2FF67E0((__int64)&v41, a2[4], a2[1], v17, v13, v14);
    if ( !v11 )
    {
      v26 = *(_DWORD *)(a1 + 8);
      v27 = v26;
      if ( v42 <= v26 )
        v27 = v42;
      v28 = 0;
      if ( v27 )
      {
        do
        {
          v29 = (_QWORD *)(v28 + *(_QWORD *)a1);
          v30 = *(_QWORD *)&v41[v28];
          v28 += 8;
          *v29 &= v30;
        }
        while ( v28 != 8LL * v27 );
      }
      for ( ; v26 != v27; *(_QWORD *)(*(_QWORD *)a1 + 8 * v31) = 0 )
        v31 = v27++;
      goto LABEL_12;
    }
    v20 = v44;
    if ( *(_DWORD *)(a1 + 64) >= v44 )
    {
LABEL_9:
      if ( v42 )
      {
        v21 = 8LL * v42;
        v22 = 0;
        do
        {
          v23 = (_QWORD *)(v22 + *(_QWORD *)a1);
          v24 = *(_QWORD *)&v41[v22];
          v22 += 8;
          *v23 |= v24;
        }
        while ( v22 != v21 );
      }
LABEL_12:
      if ( v41 != &v43 )
        _libc_free((unsigned __int64)v41);
      v11 = 0;
      goto LABEL_15;
    }
    v32 = *(_DWORD *)(a1 + 64) & 0x3F;
    if ( v32 )
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v32);
    v33 = *(unsigned int *)(a1 + 8);
    *(_DWORD *)(a1 + 64) = v20;
    v34 = (v20 + 63) >> 6;
    if ( v34 != v33 )
    {
      if ( v34 >= v33 )
      {
        v36 = v34 - v33;
        if ( v34 > *(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, s, v34, 8u, v18, v19);
          v33 = *(unsigned int *)(a1 + 8);
        }
        if ( 8 * v36 )
        {
          memset((void *)(*(_QWORD *)a1 + 8 * v33), 0, 8 * v36);
          LODWORD(v33) = *(_DWORD *)(a1 + 8);
        }
        v37 = *(_DWORD *)(a1 + 64);
        *(_DWORD *)(a1 + 8) = v36 + v33;
        v35 = v37 & 0x3F;
        if ( !v35 )
          goto LABEL_9;
LABEL_30:
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v35);
        goto LABEL_9;
      }
      *(_DWORD *)(a1 + 8) = (v20 + 63) >> 6;
    }
    v35 = v20 & 0x3F;
    if ( !v35 )
      goto LABEL_9;
    goto LABEL_30;
  }
  return a1;
}
