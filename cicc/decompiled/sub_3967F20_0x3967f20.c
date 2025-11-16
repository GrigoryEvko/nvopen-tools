// Function: sub_3967F20
// Address: 0x3967f20
//
void __fastcall sub_3967F20(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  __int64 v6; // rax
  int v7; // r14d
  _QWORD *v8; // rbx
  unsigned int v9; // eax
  __int64 v10; // rdx
  _QWORD *v11; // r13
  unsigned __int64 v12; // r14
  unsigned int v13; // ecx
  unsigned int v14; // eax
  int v15; // r14d
  unsigned int v16; // eax
  unsigned __int64 v17; // r15
  int v18; // edx
  int v19; // r13d
  unsigned int v20; // r14d
  unsigned int v21; // eax
  unsigned int v22; // eax

  v2 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 76) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 80);
    if ( (unsigned int)v3 <= 0x40 )
      goto LABEL_4;
    j___libc_free_0(*(_QWORD *)(a1 + 64));
    *(_DWORD *)(a1 + 80) = 0;
LABEL_51:
    *(_QWORD *)(a1 + 64) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 72) = 0;
    goto LABEL_7;
  }
  v13 = 4 * v2;
  a2 = 64;
  v3 = *(unsigned int *)(a1 + 80);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v13 = 64;
  if ( v13 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 64);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    goto LABEL_6;
  }
  v14 = v2 - 1;
  if ( v14 )
  {
    _BitScanReverse(&v14, v14);
    v15 = 1 << (33 - (v14 ^ 0x1F));
    if ( v15 < 64 )
      v15 = 64;
    if ( (_DWORD)v3 == v15 )
      goto LABEL_32;
  }
  else
  {
    v15 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 64));
  v16 = sub_39618B0(v15);
  *(_DWORD *)(a1 + 80) = v16;
  if ( !v16 )
    goto LABEL_51;
  *(_QWORD *)(a1 + 64) = sub_22077B0(16LL * v16);
LABEL_32:
  sub_3963260(a1 + 56);
LABEL_7:
  v6 = *(_QWORD *)(a1 + 88);
  if ( v6 != *(_QWORD *)(a1 + 96) )
    *(_QWORD *)(a1 + 96) = v6;
  v7 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v7 || *(_DWORD *)(a1 + 132) )
  {
    v8 = *(_QWORD **)(a1 + 120);
    v9 = 4 * v7;
    v10 = *(unsigned int *)(a1 + 136);
    v11 = &v8[2 * v10];
    if ( (unsigned int)(4 * v7) < 0x40 )
      v9 = 64;
    if ( (unsigned int)v10 <= v9 )
    {
      for ( ; v8 != v11; v8 += 2 )
      {
        if ( *v8 != -8 )
        {
          if ( *v8 != -16 )
          {
            v12 = v8[1];
            if ( v12 )
            {
              _libc_free(*(_QWORD *)(v12 + 48));
              _libc_free(*(_QWORD *)(v12 + 24));
              a2 = 72;
              j_j___libc_free_0(v12);
            }
          }
          *v8 = -8;
        }
      }
      goto LABEL_21;
    }
    do
    {
      if ( *v8 != -16 && *v8 != -8 )
      {
        v17 = v8[1];
        if ( v17 )
        {
          _libc_free(*(_QWORD *)(v17 + 48));
          _libc_free(*(_QWORD *)(v17 + 24));
          a2 = 72;
          j_j___libc_free_0(v17);
        }
      }
      v8 += 2;
    }
    while ( v11 != v8 );
    v18 = *(_DWORD *)(a1 + 136);
    if ( v7 )
    {
      v19 = 64;
      v20 = v7 - 1;
      if ( v20 )
      {
        _BitScanReverse(&v21, v20);
        v19 = 1 << (33 - (v21 ^ 0x1F));
        if ( v19 < 64 )
          v19 = 64;
      }
      if ( v18 == v19 )
        goto LABEL_46;
      j___libc_free_0(*(_QWORD *)(a1 + 120));
      v22 = sub_39618B0(v19);
      *(_DWORD *)(a1 + 136) = v22;
      if ( v22 )
      {
        *(_QWORD *)(a1 + 120) = sub_22077B0(16LL * v22);
LABEL_46:
        sub_39632A0(a1 + 112);
        goto LABEL_22;
      }
    }
    else
    {
      if ( !v18 )
        goto LABEL_46;
      j___libc_free_0(*(_QWORD *)(a1 + 120));
      *(_DWORD *)(a1 + 136) = 0;
    }
    *(_QWORD *)(a1 + 120) = 0;
LABEL_21:
    *(_QWORD *)(a1 + 128) = 0;
  }
LABEL_22:
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  sub_3954A10((__int64 *)a1, a2);
  sub_1C2CBE0((__int64 *)a1);
}
