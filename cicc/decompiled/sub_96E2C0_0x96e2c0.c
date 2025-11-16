// Function: sub_96E2C0
// Address: 0x96e2c0
//
unsigned __int8 *__fastcall sub_96E2C0(unsigned __int8 *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int8 *v4; // r13
  unsigned int v6; // ebx
  unsigned __int8 *v7; // r14
  unsigned int v8; // eax
  unsigned __int64 v10; // rsi
  unsigned int v11; // ebx
  bool v12; // al
  char *v13; // rbx
  char *v14; // r12
  unsigned int v15; // r15d
  char *v16; // r15
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // r14d
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v23; // [rsp+8h] [rbp-78h] BYREF
  __int64 v24; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-68h]
  char v26; // [rsp+20h] [rbp-60h] BYREF

  v4 = a1;
  v6 = *(_DWORD *)(a2 + 8);
  if ( v6 <= 0x40 )
  {
    v7 = a1;
    if ( !*(_QWORD *)a2 )
      return v7;
    v8 = *a1;
    v7 = 0;
    if ( v8 <= 8 )
      return v7;
  }
  else
  {
    v7 = a1;
    if ( v6 == (unsigned int)sub_C444A0(a2) )
      return v7;
    v8 = *a1;
    v7 = 0;
    if ( v8 <= 8 )
      return v7;
  }
  if ( v8 != 15 && v8 > 0xB && v8 != 16 )
    return v7;
  v10 = a3;
  v23 = *((_QWORD *)a1 + 1);
  sub_AE5990(&v24, a3, &v23, a2);
  v11 = *(_DWORD *)(a2 + 8);
  if ( v11 <= 0x40 )
    v12 = *(_QWORD *)a2 == 0;
  else
    v12 = v11 == (unsigned int)sub_C444A0(a2);
  v13 = (char *)v24;
  v14 = (char *)(v24 + 16LL * v25);
  if ( !v12 )
  {
LABEL_22:
    v7 = 0;
    goto LABEL_23;
  }
  v15 = *(_DWORD *)(v24 + 8);
  if ( v15 <= 0x40 )
  {
    if ( !*(_QWORD *)v24 )
    {
LABEL_10:
      v16 = v13 + 16;
      if ( v13 + 16 == v14 )
      {
        v7 = a1;
      }
      else
      {
        do
        {
          v19 = *((_DWORD *)v16 + 2);
          v20 = *(_QWORD *)v16;
          v21 = 1LL << ((unsigned __int8)v19 - 1);
          if ( v19 <= 0x40 )
          {
            v10 = *(_QWORD *)v16;
            if ( (v21 & v20) != 0 || v20 && (_BitScanReverse64(&v17, v20), 64 - ((unsigned int)v17 ^ 0x3F) > 0x1F) )
            {
LABEL_21:
              v13 = (char *)v24;
              v14 = (char *)(v24 + 16LL * v25);
              goto LABEL_22;
            }
          }
          else
          {
            if ( (*(_QWORD *)(v20 + 8LL * ((v19 - 1) >> 6)) & v21) != 0 || v19 - (unsigned int)sub_C444A0(v16) > 0x1F )
              goto LABEL_21;
            v10 = *(_QWORD *)v20;
          }
          v18 = sub_AD69F0(v4, v10);
          v4 = (unsigned __int8 *)v18;
          if ( !v18 )
            goto LABEL_21;
          v16 += 16;
        }
        while ( v14 != v16 );
        v13 = (char *)v24;
        v7 = (unsigned __int8 *)v18;
        v14 = (char *)(v24 + 16LL * v25);
      }
      goto LABEL_23;
    }
    goto LABEL_22;
  }
  v7 = 0;
  if ( v15 == (unsigned int)sub_C444A0(v24) )
    goto LABEL_10;
LABEL_23:
  if ( v13 != v14 )
  {
    do
    {
      v14 -= 16;
      if ( *((_DWORD *)v14 + 2) > 0x40u && *(_QWORD *)v14 )
        j_j___libc_free_0_0(*(_QWORD *)v14);
    }
    while ( v13 != v14 );
    v14 = (char *)v24;
  }
  if ( v14 != &v26 )
    _libc_free(v14, v10);
  return v7;
}
