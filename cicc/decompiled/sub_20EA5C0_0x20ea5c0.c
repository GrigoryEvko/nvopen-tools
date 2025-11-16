// Function: sub_20EA5C0
// Address: 0x20ea5c0
//
void __fastcall sub_20EA5C0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // r13
  __int64 v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // r14
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi

  *(_QWORD *)a1 = off_4A00A68;
  v2 = *(unsigned int *)(a1 + 360);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 344);
    v4 = v3 + 168 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
        {
          v5 = *(_QWORD *)(v3 + 88);
          if ( v5 != v3 + 104 )
            _libc_free(v5);
          if ( (*(_BYTE *)(v3 + 16) & 1) == 0 )
            break;
        }
        v3 += 168;
        if ( v4 == v3 )
          goto LABEL_9;
      }
      v6 = *(_QWORD *)(v3 + 24);
      v3 += 168;
      j___libc_free_0(v6);
    }
    while ( v4 != v3 );
  }
LABEL_9:
  j___libc_free_0(*(_QWORD *)(a1 + 344));
  v7 = *(_QWORD *)(a1 + 320);
  v8 = *(_QWORD *)(a1 + 312);
  if ( v7 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v8 + 32);
      if ( v9 != *(_QWORD *)(v8 + 24) )
        _libc_free(v9);
      v8 += 184;
    }
    while ( v7 != v8 );
    v8 = *(_QWORD *)(a1 + 312);
  }
  if ( v8 )
    j_j___libc_free_0(v8, *(_QWORD *)(a1 + 328) - v8);
  j___libc_free_0(*(_QWORD *)(a1 + 288));
  v10 = *(unsigned int *)(a1 + 272);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD *)(a1 + 256);
    v12 = v11 + 16 * v10;
    do
    {
      while ( 1 )
      {
        if ( (unsigned int)(*(_DWORD *)v11 + 0x7FFFFFFF) <= 0xFFFFFFFD )
        {
          v13 = *(unsigned __int64 **)(v11 + 8);
          if ( v13 )
            break;
        }
        v11 += 16;
        if ( v12 == v11 )
          goto LABEL_28;
      }
      sub_1DB4CE0(*(_QWORD *)(v11 + 8));
      v14 = v13[12];
      if ( v14 )
      {
        sub_20EA3F0(*(_QWORD *)(v14 + 16));
        j_j___libc_free_0(v14, 48);
      }
      v15 = v13[8];
      if ( (unsigned __int64 *)v15 != v13 + 10 )
        _libc_free(v15);
      if ( (unsigned __int64 *)*v13 != v13 + 2 )
        _libc_free(*v13);
      v11 += 16;
      j_j___libc_free_0(v13, 120);
    }
    while ( v12 != v11 );
  }
LABEL_28:
  v16 = *(_QWORD *)(a1 + 256);
  v17 = a1 + 120;
  j___libc_free_0(v16);
  v18 = *(_QWORD *)(v17 - 16);
  if ( v18 != v17 )
    _libc_free(v18);
}
