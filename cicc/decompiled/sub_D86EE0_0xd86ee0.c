// Function: sub_D86EE0
// Address: 0xd86ee0
//
__int64 __fastcall sub_D86EE0(__int64 *a1)
{
  unsigned int v1; // edx
  unsigned __int64 *v2; // r13
  int v3; // esi
  __int64 v4; // rbx
  unsigned __int64 v5; // r14
  __int64 v6; // r15
  unsigned int v7; // esi
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  bool v10; // cc
  unsigned __int64 v11; // rdi
  unsigned __int64 v13; // rdi
  int v14; // edx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  __int64 v17; // [rsp+0h] [rbp-40h]
  int v18; // [rsp+Ch] [rbp-34h]

  v1 = 0;
  v2 = (unsigned __int64 *)a1;
  v3 = *((_DWORD *)a1 + 6);
  v4 = a1[1];
  *((_DWORD *)a1 + 6) = 0;
  v5 = *a1;
  v6 = a1[2];
  v18 = v3;
  v7 = *((_DWORD *)a1 + 10);
  *((_DWORD *)a1 + 10) = 0;
  v17 = a1[4];
  while ( 1 )
  {
    v8 = *(v2 - 6);
    if ( v5 >= v8
      && (v5 != v8 || *(_QWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL) >= *(_QWORD *)(*(v2 - 5) & 0xFFFFFFFFFFFFFFF8LL)) )
    {
      break;
    }
    *v2 = v8;
    v2[1] = *(v2 - 5);
    if ( v1 > 0x40 )
    {
      v13 = v2[2];
      if ( v13 )
        j_j___libc_free_0_0(v13);
    }
    v2[2] = *(v2 - 4);
    v14 = *((_DWORD *)v2 - 6);
    *((_DWORD *)v2 - 6) = 0;
    *((_DWORD *)v2 + 6) = v14;
    v1 = 0;
    if ( *((_DWORD *)v2 + 10) > 0x40u )
    {
      v15 = v2[4];
      if ( v15 )
      {
        j_j___libc_free_0_0(v15);
        v1 = *((_DWORD *)v2 - 6);
      }
    }
    v16 = *(v2 - 2);
    v2 -= 6;
    v2[10] = v16;
    LODWORD(v16) = *((_DWORD *)v2 + 10);
    *((_DWORD *)v2 + 10) = 0;
    *((_DWORD *)v2 + 22) = v16;
  }
  *v2 = v5;
  v2[1] = v4;
  if ( v1 > 0x40 )
  {
    v9 = v2[2];
    if ( v9 )
      j_j___libc_free_0_0(v9);
  }
  v10 = *((_DWORD *)v2 + 10) <= 0x40u;
  v2[2] = v6;
  *((_DWORD *)v2 + 6) = v18;
  if ( !v10 )
  {
    v11 = v2[4];
    if ( v11 )
      j_j___libc_free_0_0(v11);
  }
  v2[4] = v17;
  *((_DWORD *)v2 + 10) = v7;
  return v7;
}
