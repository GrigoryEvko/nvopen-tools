// Function: sub_2DAD790
// Address: 0x2dad790
//
__int64 __fastcall sub_2DAD790(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // bl
  __int64 **v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  void **v9; // rax
  __int64 v10; // rsi
  _QWORD v11[18]; // [rsp+0h] [rbp-90h] BYREF

  memset(v11, 0, 0x60u);
  v11[3] = &v11[5];
  v11[4] = 0x600000000LL;
  v3 = sub_2DAD5B0((__int64)v11, a3);
  if ( (_QWORD *)v11[3] != &v11[5] )
    _libc_free(v11[3]);
  if ( !v3 )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0(v11);
  if ( HIDWORD(v11[8]) == LODWORD(v11[9]) )
  {
    if ( BYTE4(v11[3]) )
    {
      v9 = (void **)v11[1];
      v10 = v11[1] + 8LL * HIDWORD(v11[2]);
      v6 = HIDWORD(v11[2]);
      v5 = (__int64 **)v11[1];
      if ( v11[1] != v10 )
      {
        while ( *v5 != &qword_4F82400 )
        {
          if ( (__int64 **)v10 == ++v5 )
          {
LABEL_11:
            while ( *v9 != &unk_4F82408 )
            {
              if ( v5 == (__int64 **)++v9 )
                goto LABEL_16;
            }
            goto LABEL_12;
          }
        }
        goto LABEL_12;
      }
      goto LABEL_16;
    }
    if ( sub_C8CA60((__int64)v11, (__int64)&qword_4F82400) )
      goto LABEL_12;
  }
  if ( !BYTE4(v11[3]) )
  {
LABEL_18:
    sub_C8CC70((__int64)v11, (__int64)&unk_4F82408, (__int64)v5, v6, v7, v8);
    goto LABEL_12;
  }
  v9 = (void **)v11[1];
  v6 = HIDWORD(v11[2]);
  v5 = (__int64 **)(v11[1] + 8LL * HIDWORD(v11[2]));
  if ( (__int64 **)v11[1] != v5 )
    goto LABEL_11;
LABEL_16:
  if ( (unsigned int)v6 >= LODWORD(v11[2]) )
    goto LABEL_18;
  HIDWORD(v11[2]) = v6 + 1;
  *v5 = (__int64 *)&unk_4F82408;
  ++v11[0];
LABEL_12:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v11[4], (__int64)v11);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v11[10], (__int64)&v11[6]);
  if ( !BYTE4(v11[9]) )
    _libc_free(v11[7]);
  if ( !BYTE4(v11[3]) )
    _libc_free(v11[1]);
  return a1;
}
