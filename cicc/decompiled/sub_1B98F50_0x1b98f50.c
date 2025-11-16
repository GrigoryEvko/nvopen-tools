// Function: sub_1B98F50
// Address: 0x1b98f50
//
__int64 __fastcall sub_1B98F50(unsigned __int64 **a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // r13
  unsigned __int64 *v9; // r12
  unsigned __int64 *v10; // r14
  __int64 v11; // r15
  __int64 v12; // rdx
  char **v13; // rsi
  __int64 v14; // rdi
  unsigned __int64 *v15; // r15
  unsigned int v17; // [rsp+Ch] [rbp-34h]

  if ( a2 > 0xFFFFFFFF )
  {
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
LABEL_21:
    v8 = malloc(0x2FFFFFFFD0uLL);
    if ( v8 )
    {
      v17 = -1;
      goto LABEL_6;
    }
    LODWORD(v3) = -1;
    goto LABEL_24;
  }
  v3 = a2;
  v4 = sub_1454B60(*((unsigned int *)a1 + 3) + 2LL);
  if ( v4 >= a2 )
    v3 = v4;
  if ( v3 > 0xFFFFFFFF )
    goto LABEL_21;
  v17 = v3;
  v8 = malloc(48 * v3);
  if ( !v8 && (48 * v3 || (v8 = malloc(1u)) == 0) )
  {
LABEL_24:
    v8 = 0;
    sub_16BD1C0("Allocation failed", 1u);
    v17 = v3;
  }
LABEL_6:
  v9 = &(*a1)[6 * *((unsigned int *)a1 + 2)];
  if ( *a1 != v9 )
  {
    v10 = *a1;
    v11 = v8;
    do
    {
      while ( 1 )
      {
        if ( v11 )
        {
          v12 = v11 + 16;
          *(_DWORD *)(v11 + 8) = 0;
          *(_QWORD *)v11 = v11 + 16;
          *(_DWORD *)(v11 + 12) = 4;
          if ( *((_DWORD *)v10 + 2) )
            break;
        }
        v10 += 6;
        v11 += 48;
        if ( v9 == v10 )
          goto LABEL_12;
      }
      v13 = (char **)v10;
      v14 = v11;
      v10 += 6;
      v11 += 48;
      sub_1B8E3C0(v14, v13, v12, v5, v6, v7);
    }
    while ( v9 != v10 );
LABEL_12:
    v15 = *a1;
    v9 = &(*a1)[6 * *((unsigned int *)a1 + 2)];
    if ( *a1 != v9 )
    {
      do
      {
        v9 -= 6;
        if ( (unsigned __int64 *)*v9 != v9 + 2 )
          _libc_free(*v9);
      }
      while ( v9 != v15 );
      v9 = *a1;
    }
  }
  if ( v9 != (unsigned __int64 *)(a1 + 2) )
    _libc_free((unsigned __int64)v9);
  *a1 = (unsigned __int64 *)v8;
  *((_DWORD *)a1 + 3) = v17;
  return v17;
}
