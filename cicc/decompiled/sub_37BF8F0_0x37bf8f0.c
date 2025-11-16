// Function: sub_37BF8F0
// Address: 0x37bf8f0
//
void __fastcall sub_37BF8F0(__int64 a1)
{
  int v2; // ebx
  __int64 v3; // r13
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  unsigned __int64 v6; // rdi
  unsigned int v7; // eax
  int v8; // r15d
  unsigned int v9; // ebx
  unsigned int v10; // eax
  unsigned int v11; // eax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 <= 0x40 )
    {
LABEL_4:
      v4 = *(_QWORD **)(a1 + 8);
      v5 = &v4[7 * v3];
      if ( v5 == v4 )
        goto LABEL_14;
      while ( *v4 == -4096 )
      {
        if ( v4[1] == -1 && v4[2] == -1 )
        {
          v4 += 7;
          if ( v5 == v4 )
            goto LABEL_14;
        }
        else
        {
LABEL_7:
          v6 = v4[3];
          if ( (_QWORD *)v6 != v4 + 5 )
            _libc_free(v6);
LABEL_9:
          *v4 = -4096;
          v4 += 7;
          *(v4 - 6) = -1;
          *(v4 - 5) = -1;
          if ( v5 == v4 )
            goto LABEL_14;
        }
      }
      if ( *v4 == -8192 && v4[1] == -2 && v4[2] == -2 )
        goto LABEL_9;
      goto LABEL_7;
    }
    sub_37BDE90(a1);
    if ( *(_DWORD *)(a1 + 24) )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 56 * v3, 8);
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_31;
    }
LABEL_25:
    sub_37BF8A0(a1);
    return;
  }
  v7 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  if ( v7 >= (unsigned int)v3 )
    goto LABEL_4;
  v8 = 64;
  sub_37BDE90(a1);
  v9 = v2 - 1;
  if ( v9 )
  {
    _BitScanReverse(&v10, v9);
    v8 = 1 << (33 - (v10 ^ 0x1F));
    if ( v8 < 64 )
      v8 = 64;
  }
  if ( *(_DWORD *)(a1 + 24) == v8 )
    goto LABEL_25;
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 56 * v3, 8);
  v11 = sub_37B8280(v8);
  *(_DWORD *)(a1 + 24) = v11;
  if ( v11 )
  {
    *(_QWORD *)(a1 + 8) = sub_C7D670(56LL * v11, 8);
    goto LABEL_25;
  }
LABEL_31:
  *(_QWORD *)(a1 + 8) = 0;
LABEL_14:
  *(_QWORD *)(a1 + 16) = 0;
}
