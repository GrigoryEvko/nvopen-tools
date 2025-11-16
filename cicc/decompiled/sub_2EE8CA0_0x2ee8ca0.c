// Function: sub_2EE8CA0
// Address: 0x2ee8ca0
//
void __fastcall sub_2EE8CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 i, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 k; // r14
  int v11; // edx
  __int64 v12; // rsi
  int v13; // edi
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r8
  int v17; // edx
  int v18; // r9d
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r9
  __int64 *v24; // rbx
  __int64 j; // r8
  __int64 v26; // r13
  __int64 v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 *v31; // rbx
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // [rsp+0h] [rbp-E0h]
  __int64 v35; // [rsp+8h] [rbp-D8h]
  __int64 v36; // [rsp+8h] [rbp-D8h]
  __int64 v37; // [rsp+10h] [rbp-D0h]
  __int64 v38; // [rsp+10h] [rbp-D0h]
  _QWORD *v39; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+28h] [rbp-B8h]
  _QWORD v41[22]; // [rsp+30h] [rbp-B0h] BYREF

  v40 = 0x1000000000LL;
  v7 = *(int *)(a2 + 24);
  v39 = v41;
  v8 = *(_QWORD *)(a1 + 8) + 88 * v7;
  if ( *(_DWORD *)(v8 + 28) != -1 )
  {
    *(_DWORD *)(v8 + 28) = -1;
    v28 = v41;
    LODWORD(v29) = 1;
    *(_BYTE *)(v8 + 33) = 0;
    v41[0] = a2;
    LODWORD(v40) = 1;
    while ( 1 )
    {
      v30 = (unsigned int)v29;
      v29 = (unsigned int)(v29 - 1);
      a6 = v28[v30 - 1];
      LODWORD(v40) = v29;
      v31 = *(__int64 **)(a6 + 64);
      for ( i = (__int64)&v31[*(unsigned int *)(a6 + 72)]; (__int64 *)i != v31; ++v31 )
      {
        v32 = *v31;
        v33 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(*v31 + 24);
        if ( *(_DWORD *)(v33 + 28) != -1 && *(_QWORD *)(v33 + 8) == a6 )
        {
          *(_DWORD *)(v33 + 28) = -1;
          *(_BYTE *)(v33 + 33) = 0;
          if ( v29 + 1 > (unsigned __int64)HIDWORD(v40) )
          {
            v34 = a6;
            v36 = v8;
            v38 = i;
            sub_C8D5F0((__int64)&v39, v41, v29 + 1, 8u, i, a6);
            v29 = (unsigned int)v40;
            a6 = v34;
            v8 = v36;
            i = v38;
          }
          v39[v29] = v32;
          v29 = (unsigned int)(v40 + 1);
          LODWORD(v40) = v40 + 1;
        }
      }
      if ( !(_DWORD)v29 )
        break;
      v28 = v39;
    }
  }
  if ( *(_DWORD *)(v8 + 24) != -1 )
  {
    v19 = (unsigned int)v40;
    v20 = HIDWORD(v40);
    *(_BYTE *)(v8 + 32) = 0;
    *(_DWORD *)(v8 + 24) = -1;
    if ( v19 + 1 > v20 )
    {
      sub_C8D5F0((__int64)&v39, v41, v19 + 1, 8u, i, a6);
      v19 = (unsigned int)v40;
    }
    v39[v19] = a2;
    LODWORD(v21) = v40 + 1;
    LODWORD(v40) = v40 + 1;
    do
    {
      v22 = (unsigned int)v21;
      v21 = (unsigned int)(v21 - 1);
      v23 = v39[v22 - 1];
      LODWORD(v40) = v21;
      v24 = *(__int64 **)(v23 + 112);
      for ( j = (__int64)&v24[*(unsigned int *)(v23 + 120)]; (__int64 *)j != v24; ++v24 )
      {
        v26 = *v24;
        v27 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(*v24 + 24);
        if ( *(_DWORD *)(v27 + 24) != -1 && *(_QWORD *)v27 == v23 )
        {
          *(_DWORD *)(v27 + 24) = -1;
          *(_BYTE *)(v27 + 32) = 0;
          if ( v21 + 1 > (unsigned __int64)HIDWORD(v40) )
          {
            v35 = v23;
            v37 = j;
            sub_C8D5F0((__int64)&v39, v41, v21 + 1, 8u, j, v23);
            v21 = (unsigned int)v40;
            v23 = v35;
            j = v37;
          }
          v39[v21] = v26;
          v21 = (unsigned int)(v40 + 1);
          LODWORD(v40) = v40 + 1;
        }
      }
    }
    while ( (_DWORD)v21 );
  }
  v9 = *(_QWORD *)(a2 + 56);
  for ( k = a2 + 48; k != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(a1 + 400);
      v12 = *(_QWORD *)(a1 + 384);
      if ( v11 )
      {
        v13 = v11 - 1;
        v14 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v9 == *v15 )
        {
LABEL_8:
          *v15 = -8192;
          --*(_DWORD *)(a1 + 392);
          ++*(_DWORD *)(a1 + 396);
        }
        else
        {
          v17 = 1;
          while ( v16 != -4096 )
          {
            v18 = v17 + 1;
            v14 = v13 & (v17 + v14);
            v15 = (__int64 *)(v12 + 16LL * v14);
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
    while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
      v9 = *(_QWORD *)(v9 + 8);
  }
LABEL_12:
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
}
