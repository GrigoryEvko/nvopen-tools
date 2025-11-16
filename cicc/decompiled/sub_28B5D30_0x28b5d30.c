// Function: sub_28B5D30
// Address: 0x28b5d30
//
void __fastcall sub_28B5D30(_QWORD *a1)
{
  unsigned __int64 v1; // r14
  __int64 v2; // r12
  _QWORD *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned int v9; // edx
  __int64 v10; // rax
  unsigned int v11; // ecx
  __int64 v12; // rax
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 i; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]

  v1 = 0;
  v2 = 0;
  v3 = a1;
  v4 = *a1;
  v5 = a1[1];
  *a1 = 0;
  a1[1] = 0;
  v21 = v4;
  v20 = a1[2];
  a1[2] = 0;
  while ( 1 )
  {
    v6 = v21;
    v7 = *(v3 - 2);
    v8 = *(v3 - 3);
    if ( v21 == v5 )
      break;
    v9 = -1;
    do
    {
      if ( v9 > *(_DWORD *)(v6 + 92) )
        v9 = *(_DWORD *)(v6 + 92);
      v6 += 192;
    }
    while ( v5 != v6 );
    if ( v7 == v8 )
    {
      v11 = -1;
      goto LABEL_12;
    }
LABEL_8:
    v10 = *(v3 - 3);
    v11 = -1;
    do
    {
      if ( v11 > *(_DWORD *)(v10 + 92) )
        v11 = *(_DWORD *)(v10 + 92);
      v10 += 192;
    }
    while ( v7 != v10 );
LABEL_12:
    if ( v11 <= v9 )
      goto LABEL_26;
    v12 = *(v3 - 2);
    *v3 = v8;
    v13 = v1;
    *(v3 - 3) = 0;
    v3[1] = v12;
    v14 = *(v3 - 1);
    *(v3 - 2) = 0;
    v3[2] = v14;
    for ( *(v3 - 1) = 0; v2 != v13; v13 += 192LL )
    {
      if ( *(_DWORD *)(v13 + 168) > 0x40u )
      {
        v15 = *(_QWORD *)(v13 + 160);
        if ( v15 )
          j_j___libc_free_0_0(v15);
      }
      if ( *(_DWORD *)(v13 + 128) > 0x40u )
      {
        v16 = *(_QWORD *)(v13 + 120);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      if ( (*(_BYTE *)(v13 + 16) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v13 + 24), 8LL * *(unsigned int *)(v13 + 32), 8);
    }
    if ( v1 )
      j_j___libc_free_0(v1);
    v1 = *(v3 - 3);
    v3 -= 3;
    v2 = v3[1];
  }
  if ( v7 != v8 )
  {
    v9 = -1;
    goto LABEL_8;
  }
LABEL_26:
  v3[1] = v5;
  *v3 = v21;
  v3[2] = v20;
  for ( i = v1; v2 != i; i += 192LL )
  {
    if ( *(_DWORD *)(i + 168) > 0x40u )
    {
      v18 = *(_QWORD *)(i + 160);
      if ( v18 )
        j_j___libc_free_0_0(v18);
    }
    if ( *(_DWORD *)(i + 128) > 0x40u )
    {
      v19 = *(_QWORD *)(i + 120);
      if ( v19 )
        j_j___libc_free_0_0(v19);
    }
    if ( (*(_BYTE *)(i + 16) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(i + 24), 8LL * *(unsigned int *)(i + 32), 8);
  }
  if ( v1 )
    j_j___libc_free_0(v1);
}
