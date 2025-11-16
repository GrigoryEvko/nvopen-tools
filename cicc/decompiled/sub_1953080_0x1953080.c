// Function: sub_1953080
// Address: 0x1953080
//
void __fastcall sub_1953080(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r14
  _BYTE *v4; // rbx
  __int64 *v5; // r8
  __int64 *v6; // r9
  __int64 v7; // rsi
  __int64 *v8; // rdi
  unsigned int v9; // r10d
  __int64 *v10; // rax
  __int64 *v11; // rcx
  _BYTE *v12; // [rsp+0h] [rbp-240h] BYREF
  __int64 v13; // [rsp+8h] [rbp-238h]
  _BYTE v14[560]; // [rsp+10h] [rbp-230h] BYREF

  v12 = v14;
  v13 = 0x2000000000LL;
  sub_137D9B0(a2, (__int64)&v12);
  v3 = (unsigned __int64)v12;
  v4 = &v12[16 * (unsigned int)v13];
  if ( v4 == v12 )
    goto LABEL_15;
  v5 = *(__int64 **)(a1 + 72);
  v6 = *(__int64 **)(a1 + 64);
  do
  {
LABEL_5:
    v7 = *(_QWORD *)(v3 + 8);
    if ( v5 != v6 )
    {
LABEL_3:
      sub_16CCBA0(a1 + 56, v7);
      v5 = *(__int64 **)(a1 + 72);
      v6 = *(__int64 **)(a1 + 64);
      goto LABEL_4;
    }
    v8 = &v5[*(unsigned int *)(a1 + 84)];
    v9 = *(_DWORD *)(a1 + 84);
    if ( v8 == v5 )
    {
LABEL_18:
      if ( v9 >= *(_DWORD *)(a1 + 80) )
        goto LABEL_3;
      *(_DWORD *)(a1 + 84) = v9 + 1;
      *v8 = v7;
      v6 = *(__int64 **)(a1 + 64);
      ++*(_QWORD *)(a1 + 56);
      v5 = *(__int64 **)(a1 + 72);
    }
    else
    {
      v10 = v5;
      v11 = 0;
      while ( v7 != *v10 )
      {
        if ( *v10 == -2 )
          v11 = v10;
        if ( v8 == ++v10 )
        {
          if ( !v11 )
            goto LABEL_18;
          v3 += 16LL;
          *v11 = v7;
          v5 = *(__int64 **)(a1 + 72);
          --*(_DWORD *)(a1 + 88);
          v6 = *(__int64 **)(a1 + 64);
          ++*(_QWORD *)(a1 + 56);
          if ( (_BYTE *)v3 != v4 )
            goto LABEL_5;
          goto LABEL_14;
        }
      }
    }
LABEL_4:
    v3 += 16LL;
  }
  while ( (_BYTE *)v3 != v4 );
LABEL_14:
  v3 = (unsigned __int64)v12;
LABEL_15:
  if ( (_BYTE *)v3 != v14 )
    _libc_free(v3);
}
