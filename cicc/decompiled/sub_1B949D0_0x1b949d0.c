// Function: sub_1B949D0
// Address: 0x1b949d0
//
void __fastcall sub_1B949D0(__int64 a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r14
  __int64 v4; // r12
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  __int64 v16; // r14
  unsigned __int64 v17; // rdi

  if ( *(_QWORD *)a1 )
    sub_1BE4260();
  if ( *(_DWORD *)(a1 + 296) )
  {
    v13 = *(_QWORD **)(a1 + 288);
    v14 = &v13[2 * *(unsigned int *)(a1 + 304)];
    if ( v13 != v14 )
    {
      while ( 1 )
      {
        v15 = v13;
        if ( *v13 != -16 && *v13 != -8 )
          break;
        v13 += 2;
        if ( v14 == v13 )
          goto LABEL_4;
      }
      while ( v14 != v15 )
      {
        v16 = v15[1];
        if ( v16 )
        {
          v17 = *(_QWORD *)(v16 + 8);
          if ( v17 != v16 + 24 )
            _libc_free(v17);
          j_j___libc_free_0(v16, 40);
        }
        v15 += 2;
        if ( v15 == v14 )
          break;
        while ( *v15 == -8 || *v15 == -16 )
        {
          v15 += 2;
          if ( v14 == v15 )
          {
            v2 = *(_QWORD **)(a1 + 128);
            if ( v2 != *(_QWORD **)(a1 + 120) )
              goto LABEL_5;
            goto LABEL_42;
          }
        }
      }
    }
  }
LABEL_4:
  v2 = *(_QWORD **)(a1 + 128);
  if ( v2 == *(_QWORD **)(a1 + 120) )
LABEL_42:
    v3 = &v2[*(unsigned int *)(a1 + 140)];
  else
LABEL_5:
    v3 = &v2[*(unsigned int *)(a1 + 136)];
  if ( v2 != v3 )
  {
    while ( 1 )
    {
      v4 = *v2;
      v5 = v2;
      if ( *v2 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v3 == ++v2 )
        goto LABEL_9;
    }
    if ( v3 != v2 )
    {
      do
      {
        if ( v4 )
        {
          v11 = *(_QWORD *)(v4 + 8);
          if ( v11 != v4 + 24 )
            _libc_free(v11);
          j_j___libc_free_0(v4, 40);
        }
        v12 = v5 + 1;
        if ( v5 + 1 == v3 )
          break;
        v4 = *v12;
        for ( ++v5; *v12 >= 0xFFFFFFFFFFFFFFFELL; v5 = v12 )
        {
          if ( v3 == ++v12 )
            goto LABEL_9;
          v4 = *v12;
        }
      }
      while ( v5 != v3 );
    }
  }
LABEL_9:
  sub_1B945B0(a1 + 312);
  j___libc_free_0(*(_QWORD *)(a1 + 288));
  v6 = *(_QWORD *)(a1 + 128);
  if ( v6 != *(_QWORD *)(a1 + 120) )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 80);
  if ( v7 != a1 + 96 )
    j_j___libc_free_0(v7, *(_QWORD *)(a1 + 96) + 1LL);
  v8 = *(_QWORD *)(a1 + 48);
  v9 = a1 + 24;
  sub_1B8EF70(v8);
  v10 = *(_QWORD *)(v9 - 16);
  if ( v10 != v9 )
    _libc_free(v10);
}
