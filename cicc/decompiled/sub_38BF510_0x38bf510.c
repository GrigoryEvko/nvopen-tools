// Function: sub_38BF510
// Address: 0x38bf510
//
__int64 __fastcall sub_38BF510(__int64 a1, __int64 a2)
{
  bool v2; // zf
  char v3; // al
  const char *v4; // rbx
  size_t v5; // r13
  unsigned __int8 *v6; // r14
  unsigned int v7; // r8d
  int v8; // r9d
  _QWORD *v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v13; // rax
  unsigned int v14; // r8d
  _QWORD *v15; // rcx
  _QWORD *v16; // rbx
  __int64 *v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // [rsp+8h] [rbp-D8h]
  unsigned int v20; // [rsp+14h] [rbp-CCh]
  void *src; // [rsp+20h] [rbp-C0h] BYREF
  size_t n; // [rsp+28h] [rbp-B8h]
  _BYTE v23[176]; // [rsp+30h] [rbp-B0h] BYREF

  v2 = *(_BYTE *)(a2 + 17) == 1;
  src = v23;
  n = 0x8000000000LL;
  if ( v2 )
  {
    v3 = *(_BYTE *)(a2 + 16);
    if ( v3 == 1 )
    {
      v5 = 0;
      v6 = 0;
    }
    else
    {
      v4 = *(const char **)a2;
      switch ( v3 )
      {
        case 3:
          v5 = 0;
          if ( v4 )
            v5 = strlen(*(const char **)a2);
          v6 = (unsigned __int8 *)v4;
          break;
        case 4:
        case 5:
          v6 = *(unsigned __int8 **)v4;
          v5 = *((_QWORD *)v4 + 1);
          break;
        case 6:
          v5 = *((unsigned int *)v4 + 2);
          v6 = *(unsigned __int8 **)v4;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a2, (__int64)&src);
    v5 = (unsigned int)n;
    v6 = (unsigned __int8 *)src;
  }
  v7 = sub_16D19C0(a1 + 568, v6, v5);
  v9 = (_QWORD *)(*(_QWORD *)(a1 + 568) + 8LL * v7);
  v10 = *v9;
  if ( *v9 )
  {
    if ( v10 != -8 )
      goto LABEL_7;
    --*(_DWORD *)(a1 + 584);
  }
  v19 = v9;
  v20 = v7;
  v13 = sub_145CBF0(*(__int64 **)(a1 + 592), v5 + 17, 8);
  v14 = v20;
  v15 = v19;
  v16 = (_QWORD *)v13;
  if ( v5 + 1 > 1 )
  {
    memcpy((void *)(v13 + 16), v6, v5);
    v15 = v19;
    v14 = v20;
  }
  *((_BYTE *)v16 + v5 + 16) = 0;
  *v16 = v5;
  v16[1] = 0;
  *v15 = v16;
  ++*(_DWORD *)(a1 + 580);
  v17 = (__int64 *)(*(_QWORD *)(a1 + 568) + 8LL * (unsigned int)sub_16D1CD0(a1 + 568, v14));
  v10 = *v17;
  if ( *v17 && v10 != -8 )
  {
LABEL_7:
    v11 = *(_QWORD *)(v10 + 8);
    if ( v11 )
      goto LABEL_8;
    goto LABEL_19;
  }
  do
  {
    do
    {
      v10 = v17[1];
      ++v17;
    }
    while ( v10 == -8 );
  }
  while ( !v10 );
  v11 = *(_QWORD *)(v10 + 8);
  if ( !v11 )
  {
LABEL_19:
    v18 = sub_38BEE30(a1, v6, v5, 0, 0, v8);
    *(_QWORD *)(v10 + 8) = v18;
    v11 = v18;
  }
LABEL_8:
  if ( src != v23 )
    _libc_free((unsigned __int64)src);
  return v11;
}
