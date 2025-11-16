// Function: sub_15CED80
// Address: 0x15ced80
//
void __fastcall sub_15CED80(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  __int64 v4; // rdx
  _QWORD *v5; // r13
  unsigned int v6; // eax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  int v9; // edx
  int v10; // ebx
  unsigned int v11; // r14d
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // [rsp+8h] [rbp-28h] BYREF
  _BYTE v15[32]; // [rsp+10h] [rbp-20h] BYREF

  v14 = 0;
  sub_15CBD60(a1, (char *)&v14, v15);
  v2 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  if ( v2 || *(_DWORD *)(a1 + 44) )
  {
    v3 = *(_QWORD **)(a1 + 32);
    v4 = *(unsigned int *)(a1 + 48);
    v5 = &v3[9 * v4];
    v6 = 4 * v2;
    if ( (unsigned int)(4 * v2) < 0x40 )
      v6 = 64;
    if ( (unsigned int)v4 <= v6 )
    {
      for ( ; v3 != v5; v3 += 9 )
      {
        if ( *v3 != -8 )
        {
          if ( *v3 != -16 )
          {
            v7 = v3[5];
            if ( (_QWORD *)v7 != v3 + 7 )
              _libc_free(v7);
          }
          *v3 = -8;
        }
      }
      goto LABEL_13;
    }
    do
    {
      if ( *v3 != -8 && *v3 != -16 )
      {
        v8 = v3[5];
        if ( (_QWORD *)v8 != v3 + 7 )
          _libc_free(v8);
      }
      v3 += 9;
    }
    while ( v3 != v5 );
    v9 = *(_DWORD *)(a1 + 48);
    if ( v2 )
    {
      v10 = 64;
      v11 = v2 - 1;
      if ( v11 )
      {
        _BitScanReverse(&v12, v11);
        v10 = 1 << (33 - (v12 ^ 0x1F));
        if ( v10 < 64 )
          v10 = 64;
      }
      if ( v10 == v9 )
        goto LABEL_28;
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      v13 = sub_1454B60(4 * v10 / 3u + 1);
      *(_DWORD *)(a1 + 48) = v13;
      if ( v13 )
      {
        *(_QWORD *)(a1 + 32) = sub_22077B0(72LL * v13);
LABEL_28:
        sub_15CED40(a1 + 24);
        return;
      }
    }
    else
    {
      if ( !v9 )
        goto LABEL_28;
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      *(_DWORD *)(a1 + 48) = 0;
    }
    *(_QWORD *)(a1 + 32) = 0;
LABEL_13:
    *(_QWORD *)(a1 + 40) = 0;
  }
}
