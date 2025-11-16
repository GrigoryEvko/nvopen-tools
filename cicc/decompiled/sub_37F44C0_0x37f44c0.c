// Function: sub_37F44C0
// Address: 0x37f44c0
//
__int64 __fastcall sub_37F44C0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rdi
  __int64 *v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rax
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v19; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = &unk_4A3D920;
  v2 = *(unsigned int *)(a1 + 632);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 616);
    v4 = v3 + 72 * v2;
    while ( 1 )
    {
      while ( *(_DWORD *)v3 == -1 )
      {
        if ( *(_DWORD *)(v3 + 4) != 0x7FFFFFFF )
          goto LABEL_4;
        v3 += 72;
        if ( v4 == v3 )
        {
LABEL_10:
          v2 = *(unsigned int *)(a1 + 632);
          goto LABEL_11;
        }
      }
      if ( *(_DWORD *)v3 != -2 || *(_DWORD *)(v3 + 4) != 0x80000000 )
      {
LABEL_4:
        v5 = *(_QWORD *)(v3 + 8);
        if ( v5 != v3 + 24 )
          _libc_free(v5);
      }
      v3 += 72;
      if ( v4 == v3 )
        goto LABEL_10;
    }
  }
LABEL_11:
  sub_C7D6A0(*(_QWORD *)(a1 + 616), 72 * v2, 8);
  v19 = *(unsigned __int64 **)(a1 + 496);
  v6 = &v19[3 * *(unsigned int *)(a1 + 504)];
  if ( v19 != v6 )
  {
    do
    {
      v7 = *(v6 - 3);
      v8 = (__int64 *)*(v6 - 2);
      v6 -= 3;
      v9 = (__int64 *)v7;
      if ( v8 != (__int64 *)v7 )
      {
        do
        {
          v10 = *v9;
          if ( *v9 )
          {
            if ( (v10 & 1) != 0 )
            {
              v11 = (unsigned __int64 *)(v10 & 0xFFFFFFFFFFFFFFFELL);
              v12 = (unsigned __int64)v11;
              if ( v11 )
              {
                if ( (unsigned __int64 *)*v11 != v11 + 2 )
                  _libc_free(*v11);
                j_j___libc_free_0(v12);
              }
            }
          }
          ++v9;
        }
        while ( v8 != v9 );
        v7 = *v6;
      }
      if ( v7 )
        j_j___libc_free_0(v7);
    }
    while ( v19 != v6 );
    v6 = *(unsigned __int64 **)(a1 + 496);
  }
  if ( v6 != (unsigned __int64 *)(a1 + 512) )
    _libc_free((unsigned __int64)v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 472), 16LL * *(unsigned int *)(a1 + 488), 8);
  v13 = *(_QWORD *)(a1 + 344);
  v14 = v13 + 24LL * *(unsigned int *)(a1 + 352);
  if ( v13 != v14 )
  {
    do
    {
      v15 = *(_QWORD *)(v14 - 24);
      v14 -= 24LL;
      if ( v15 )
        j_j___libc_free_0(v15);
    }
    while ( v13 != v14 );
    v14 = *(_QWORD *)(a1 + 344);
  }
  if ( v14 != a1 + 360 )
    _libc_free(v14);
  v16 = *(_QWORD *)(a1 + 320);
  if ( v16 )
    j_j___libc_free_0(v16);
  v17 = *(_QWORD *)(a1 + 224);
  if ( v17 != a1 + 240 )
    _libc_free(v17);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
