// Function: sub_37BA230
// Address: 0x37ba230
//
__int64 __fastcall sub_37BA230(__int64 a1, unsigned int a2)
{
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // r8
  unsigned int v5; // r12d
  unsigned int v6; // r15d
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  int v9; // edi
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  __int64 v16; // r9
  int v17; // r15d
  __int64 v18; // r9
  _DWORD *v19; // rdx
  _DWORD *v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // r10
  __int64 *v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 40);
  v4 = (unsigned int)(v3 + 1);
  v5 = *(_DWORD *)(a1 + 40);
  v6 = v4;
  if ( (unsigned int)v3 < (unsigned int)v4 )
  {
    v16 = *(_QWORD *)(a1 + 48);
    if ( v3 != v4 )
    {
      if ( v3 <= v4 )
      {
        v21 = *(unsigned int *)(a1 + 40);
        v22 = v4 - v3;
        if ( v4 > *(unsigned int *)(a1 + 44) )
        {
          v25 = *(_QWORD *)(a1 + 48);
          v26 = v4 - v3;
          sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), (unsigned int)(v3 + 1), 8u, v4, v16);
          v21 = *(unsigned int *)(a1 + 40);
          v16 = v25;
          v22 = v26;
          v4 = (unsigned int)(v3 + 1);
        }
        v23 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8 * v21);
        v24 = v22;
        do
        {
          if ( v23 )
            *v23 = v16;
          ++v23;
          --v24;
        }
        while ( v24 );
        *(_DWORD *)(a1 + 40) += v22;
      }
      else
      {
        *(_DWORD *)(a1 + 40) = v4;
      }
    }
  }
  v7 = *(unsigned int *)(a1 + 96);
  if ( (unsigned int)v7 < v6 && v4 != v7 )
  {
    if ( v4 >= v7 )
    {
      v17 = *(_DWORD *)(a1 + 104);
      v18 = v4 - v7;
      if ( v4 > *(unsigned int *)(a1 + 100) )
      {
        v27 = v4 - v7;
        sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v4, 4u, v4, v18);
        v7 = *(unsigned int *)(a1 + 96);
        v18 = v27;
      }
      v19 = (_DWORD *)(*(_QWORD *)(a1 + 88) + 4 * v7);
      v20 = &v19[v18];
      if ( v19 != v20 )
      {
        do
          *v19++ = v17;
        while ( v20 != v19 );
        LODWORD(v7) = *(_DWORD *)(a1 + 96);
      }
      *(_DWORD *)(a1 + 96) = v18 + v7;
    }
    else
    {
      *(_DWORD *)(a1 + 96) = v6;
    }
  }
  v8 = *(_QWORD *)(a1 + 296);
  v9 = *(_DWORD *)(a1 + 280);
  v10 = v8 + 16LL * *(unsigned int *)(a1 + 304);
  if ( v8 == v10 )
  {
LABEL_8:
    v12 = 0;
  }
  else
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v10 - 16) + 24LL) + 4LL * (a2 >> 5));
      if ( !_bittest(&v11, a2 & 0x1F) )
        break;
      v10 -= 16;
      if ( v8 == v10 )
        goto LABEL_8;
    }
    v12 = *(_DWORD *)(v10 - 8) & 0xFFFFF;
  }
  v13 = *(_QWORD *)(a1 + 32) + 8 * v3;
  v14 = *(_QWORD *)v13 & 0xFFFFFF0000000000LL | (v12 << 20) | v9 & 0xFFFFF;
  *(_QWORD *)v13 = v14;
  *(_DWORD *)(v13 + 4) = BYTE4(v14) | (v5 << 8);
  *(_DWORD *)(*(_QWORD *)(a1 + 88) + 4 * v3) = a2;
  return v5;
}
