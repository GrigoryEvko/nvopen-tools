// Function: sub_2E96F10
// Address: 0x2e96f10
//
__int64 __fastcall sub_2E96F10(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 v4; // r15
  unsigned __int64 v5; // r12
  char v6; // bl
  __int64 v7; // r15
  unsigned __int64 v8; // r12
  __int64 **v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  void **v14; // rax
  __int64 v15; // rsi
  __int128 v16; // [rsp+0h] [rbp-90h] BYREF
  unsigned __int64 v17; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+18h] [rbp-78h]
  __int64 v19; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v20; // [rsp+28h] [rbp-68h]
  __int64 v21; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v22; // [rsp+38h] [rbp-58h]
  int v23; // [rsp+44h] [rbp-4Ch]
  int v24; // [rsp+48h] [rbp-48h]
  char v25; // [rsp+4Ch] [rbp-44h]
  _BYTE v26[64]; // [rsp+50h] [rbp-40h] BYREF

  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v16 = 0;
  v3 = sub_2E95AB0(&v16, a3);
  v4 = v21;
  v5 = v20;
  v6 = v3;
  if ( v21 != v20 )
  {
    do
    {
      if ( (*(_BYTE *)(v5 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v5 + 16), 16LL * *(unsigned int *)(v5 + 24), 8);
      v5 += 80LL;
    }
    while ( v4 != v5 );
    v5 = v20;
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  v7 = v18;
  v8 = v17;
  if ( v18 != v17 )
  {
    do
    {
      if ( (*(_BYTE *)(v8 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v8 + 16), 16LL * *(unsigned int *)(v8 + 24), 8);
      v8 += 80LL;
    }
    while ( v7 != v8 );
    v8 = v17;
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  if ( !v6 )
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
  sub_2EAFFB0(&v16);
  if ( v23 == v24 )
  {
    if ( BYTE4(v18) )
    {
      v14 = (void **)*((_QWORD *)&v16 + 1);
      v15 = *((_QWORD *)&v16 + 1) + 8LL * HIDWORD(v17);
      v11 = HIDWORD(v17);
      v10 = (__int64 **)*((_QWORD *)&v16 + 1);
      if ( *((_QWORD *)&v16 + 1) != v15 )
      {
        while ( *v10 != &qword_4F82400 )
        {
          if ( (__int64 **)v15 == ++v10 )
          {
LABEL_23:
            while ( *v14 != &unk_4F82408 )
            {
              if ( ++v14 == (void **)v10 )
                goto LABEL_34;
            }
            goto LABEL_24;
          }
        }
        goto LABEL_24;
      }
      goto LABEL_34;
    }
    if ( sub_C8CA60((__int64)&v16, (__int64)&qword_4F82400) )
      goto LABEL_24;
  }
  if ( !BYTE4(v18) )
  {
LABEL_36:
    sub_C8CC70((__int64)&v16, (__int64)&unk_4F82408, (__int64)v10, v11, v12, v13);
    goto LABEL_24;
  }
  v14 = (void **)*((_QWORD *)&v16 + 1);
  v11 = HIDWORD(v17);
  v10 = (__int64 **)(*((_QWORD *)&v16 + 1) + 8LL * HIDWORD(v17));
  if ( v10 != *((__int64 ***)&v16 + 1) )
    goto LABEL_23;
LABEL_34:
  if ( (unsigned int)v17 <= (unsigned int)v11 )
    goto LABEL_36;
  HIDWORD(v17) = v11 + 1;
  *v10 = (__int64 *)&unk_4F82408;
  *(_QWORD *)&v16 = v16 + 1;
LABEL_24:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v19, (__int64)&v16);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v26, (__int64)&v21);
  if ( !v25 )
    _libc_free(v22);
  if ( !BYTE4(v18) )
    _libc_free(*((unsigned __int64 *)&v16 + 1));
  return a1;
}
