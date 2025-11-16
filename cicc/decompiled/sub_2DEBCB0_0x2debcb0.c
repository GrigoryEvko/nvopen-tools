// Function: sub_2DEBCB0
// Address: 0x2debcb0
//
__int64 __fastcall sub_2DEBCB0(unsigned int *a1, int *a2)
{
  unsigned int v2; // r15d
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  const void **v6; // r14
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  unsigned int v11; // r13d
  unsigned int v12; // eax
  int v13; // ebx
  __int64 v14; // r12
  __int64 v15; // rdx
  const void **v16; // r13
  bool v17; // al
  __int64 v18; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v19; // [rsp+10h] [rbp-E0h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v21; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-C8h]
  _QWORD v23[24]; // [rsp+30h] [rbp-C0h] BYREF

  v2 = a1[34];
  if ( v2 != a2[34] )
    goto LABEL_5;
  v3 = *((_QWORD *)a1 + 1);
  v4 = *((_QWORD *)a2 + 1);
  if ( v3 )
  {
    if ( v4 != v3 )
      goto LABEL_5;
    v5 = a1[6];
    if ( v5 != a2[6] )
      goto LABEL_5;
    v14 = *((_QWORD *)a1 + 2);
    v15 = v14 + 24 * v5;
    if ( v14 != v15 )
    {
      v16 = (const void **)(*((_QWORD *)a2 + 2) + 8LL);
      while ( *(_DWORD *)v14 == *((_DWORD *)v16 - 2) )
      {
        if ( *(_DWORD *)(v14 + 16) <= 0x40u )
        {
          if ( *(const void **)(v14 + 8) != *v16 )
            goto LABEL_5;
        }
        else
        {
          v18 = v15;
          v17 = sub_C43C50(v14 + 8, v16);
          v15 = v18;
          if ( !v17 )
            goto LABEL_5;
        }
        v14 += 24;
        v16 += 3;
        if ( v15 == v14 )
          goto LABEL_16;
      }
      goto LABEL_5;
    }
  }
  else if ( v4 )
  {
LABEL_5:
    LODWORD(v6) = 0;
    memset(v23, 0, 0x90u);
    LODWORD(v23[0]) = -1;
    v23[2] = &v23[4];
    HIDWORD(v23[3]) = 4;
    goto LABEL_6;
  }
LABEL_16:
  v11 = *a2;
  v20 = v2;
  if ( *a1 >= v11 )
    v11 = *a1;
  v6 = (const void **)&v19;
  if ( v2 > 0x40 )
    sub_C43780((__int64)&v19, (const void **)a1 + 16);
  else
    v19 = *((_QWORD *)a1 + 16);
  sub_C46B40((__int64)&v19, (__int64 *)a2 + 16);
  v12 = v20;
  LODWORD(v23[0]) = v11;
  v20 = 0;
  v22 = v12;
  v21 = v19;
  v23[1] = 0;
  v23[2] = &v23[4];
  v23[3] = 0x400000000LL;
  LODWORD(v23[17]) = v12;
  if ( v12 > 0x40 )
  {
    sub_C43780((__int64)&v23[16], (const void **)&v21);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
  }
  else
  {
    v23[16] = v19;
  }
  v13 = v23[17];
  if ( LODWORD(v23[0]) || v23[1] )
  {
    LODWORD(v6) = 0;
    if ( LODWORD(v23[17]) <= 0x40 )
      goto LABEL_6;
  }
  else
  {
    if ( LODWORD(v23[17]) <= 0x40 )
    {
      LOBYTE(v6) = v23[16] == 0;
      goto LABEL_6;
    }
    LOBYTE(v6) = v13 == (unsigned int)sub_C444A0((__int64)&v23[16]);
  }
  if ( v23[16] )
    j_j___libc_free_0_0(v23[16]);
LABEL_6:
  v7 = v23[2];
  v8 = v23[2] + 24LL * LODWORD(v23[3]);
  if ( v23[2] != v8 )
  {
    do
    {
      v8 -= 24LL;
      if ( *(_DWORD *)(v8 + 16) > 0x40u )
      {
        v9 = *(_QWORD *)(v8 + 8);
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
    }
    while ( v7 != v8 );
    v8 = v23[2];
  }
  if ( (_QWORD *)v8 != &v23[4] )
    _libc_free(v8);
  return (unsigned int)v6;
}
