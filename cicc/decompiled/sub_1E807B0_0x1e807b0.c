// Function: sub_1E807B0
// Address: 0x1e807b0
//
void __fastcall sub_1E807B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 k; // r14
  int v11; // edx
  int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r8
  int v17; // edx
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 *v23; // r8
  __int64 *j; // rbx
  __int64 v25; // r13
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 *i; // rbx
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-E0h]
  __int64 v34; // [rsp+8h] [rbp-D8h]
  __int64 v35; // [rsp+8h] [rbp-D8h]
  __int64 *v36; // [rsp+10h] [rbp-D0h]
  __int64 *v37; // [rsp+10h] [rbp-D0h]
  _QWORD *v38; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+28h] [rbp-B8h]
  _QWORD v40[22]; // [rsp+30h] [rbp-B0h] BYREF

  v39 = 0x1000000000LL;
  v7 = *(int *)(a2 + 48);
  v38 = v40;
  v8 = *(_QWORD *)(a1 + 8) + 88 * v7;
  if ( *(_DWORD *)(v8 + 28) != -1 )
  {
    *(_DWORD *)(v8 + 28) = -1;
    v27 = v40;
    LODWORD(v28) = 1;
    *(_BYTE *)(v8 + 33) = 0;
    v40[0] = a2;
    LODWORD(v39) = 1;
    while ( 1 )
    {
      v29 = (unsigned int)v28;
      v28 = (unsigned int)(v28 - 1);
      a6 = v27[v29 - 1];
      LODWORD(v39) = v28;
      a5 = *(__int64 **)(a6 + 72);
      for ( i = *(__int64 **)(a6 + 64); a5 != i; ++i )
      {
        v31 = *i;
        v32 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(*i + 48);
        if ( *(_DWORD *)(v32 + 28) != -1 && *(_QWORD *)(v32 + 8) == a6 )
        {
          *(_DWORD *)(v32 + 28) = -1;
          *(_BYTE *)(v32 + 33) = 0;
          if ( HIDWORD(v39) <= (unsigned int)v28 )
          {
            v33 = a6;
            v35 = v8;
            v37 = a5;
            sub_16CD150((__int64)&v38, v40, 0, 8, (int)a5, a6);
            v28 = (unsigned int)v39;
            a6 = v33;
            v8 = v35;
            a5 = v37;
          }
          v38[v28] = v31;
          v28 = (unsigned int)(v39 + 1);
          LODWORD(v39) = v39 + 1;
        }
      }
      if ( !(_DWORD)v28 )
        break;
      v27 = v38;
    }
  }
  if ( *(_DWORD *)(v8 + 24) != -1 )
  {
    *(_DWORD *)(v8 + 24) = -1;
    v19 = (unsigned int)v39;
    *(_BYTE *)(v8 + 32) = 0;
    if ( (unsigned int)v19 >= HIDWORD(v39) )
    {
      sub_16CD150((__int64)&v38, v40, 0, 8, (int)a5, a6);
      v19 = (unsigned int)v39;
    }
    v38[v19] = a2;
    LODWORD(v20) = v39 + 1;
    LODWORD(v39) = v39 + 1;
    do
    {
      v21 = (unsigned int)v20;
      v20 = (unsigned int)(v20 - 1);
      v22 = v38[v21 - 1];
      LODWORD(v39) = v20;
      v23 = *(__int64 **)(v22 + 96);
      for ( j = *(__int64 **)(v22 + 88); v23 != j; ++j )
      {
        v25 = *j;
        v26 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(*j + 48);
        if ( *(_DWORD *)(v26 + 24) != -1 && *(_QWORD *)v26 == v22 )
        {
          *(_DWORD *)(v26 + 24) = -1;
          *(_BYTE *)(v26 + 32) = 0;
          if ( HIDWORD(v39) <= (unsigned int)v20 )
          {
            v34 = v22;
            v36 = v23;
            sub_16CD150((__int64)&v38, v40, 0, 8, (int)v23, v22);
            v20 = (unsigned int)v39;
            v22 = v34;
            v23 = v36;
          }
          v38[v20] = v25;
          v20 = (unsigned int)(v39 + 1);
          LODWORD(v39) = v39 + 1;
        }
      }
    }
    while ( (_DWORD)v20 );
  }
  v9 = *(_QWORD *)(a2 + 32);
  for ( k = a2 + 24; k != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(a1 + 400);
      if ( v11 )
      {
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 384);
        v14 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v15 = (__int64 *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( *v15 == v9 )
        {
LABEL_8:
          *v15 = -16;
          --*(_DWORD *)(a1 + 392);
          ++*(_DWORD *)(a1 + 396);
        }
        else
        {
          v17 = 1;
          while ( v16 != -8 )
          {
            v18 = v17 + 1;
            v14 = v12 & (v17 + v14);
            v15 = (__int64 *)(v13 + 16LL * v14);
            v16 = *v15;
            if ( v9 == *v15 )
              goto LABEL_8;
            v17 = v18;
          }
        }
      }
      if ( !v9 )
        BUG();
      if ( (*(_BYTE *)v9 & 4) == 0 )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( k == v9 )
        goto LABEL_12;
    }
    while ( (*(_BYTE *)(v9 + 46) & 8) != 0 )
      v9 = *(_QWORD *)(v9 + 8);
  }
LABEL_12:
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
}
