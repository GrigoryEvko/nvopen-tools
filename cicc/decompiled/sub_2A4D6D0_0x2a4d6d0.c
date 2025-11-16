// Function: sub_2A4D6D0
// Address: 0x2a4d6d0
//
__int64 __fastcall sub_2A4D6D0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r14d
  unsigned __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 *v9; // r14
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = sub_B43CA0(a1);
  v2 = *(unsigned __int8 *)(v1 + 872);
  if ( !(_BYTE)v2 )
    goto LABEL_2;
  v7 = v1;
  sub_AE7690((unsigned __int64 *)&v10, a1);
  v8 = v10;
  if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    LODWORD(v4) = v2;
    if ( !v10 || !((unsigned __int64)v10 >> 2) )
      return (unsigned int)v4;
    goto LABEL_8;
  }
  if ( ((v10 >> 2) & 1) != 0 )
  {
    if ( !*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 8) )
    {
      LODWORD(v4) = (v10 >> 2) & 1;
LABEL_29:
      v9 = (unsigned __int64 *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_8;
      goto LABEL_16;
    }
    LODWORD(v4) = 0;
    v9 = (unsigned __int64 *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
    if ( *(_BYTE *)(v7 + 872) )
    {
LABEL_16:
      if ( (unsigned __int64 *)*v9 != v9 + 2 )
        _libc_free(*v9);
      j_j___libc_free_0((unsigned __int64)v9);
      goto LABEL_8;
    }
    v2 = (v10 >> 2) & 1;
LABEL_2:
    sub_AE74C0(v11, a1);
    v3 = v11[0] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v11[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v4 = ((__int64)v11[0] >> 2) & 1;
      if ( (_DWORD)v4 )
      {
        LOBYTE(v4) = *(_DWORD *)(v3 + 8) == 0;
        if ( *(_QWORD *)v3 != v3 + 16 )
          _libc_free(*(_QWORD *)v3);
        j_j___libc_free_0(v3);
      }
      if ( !(_BYTE)v2 || (v8 = v10) == 0 )
      {
LABEL_8:
        if ( (_BYTE)v4 )
          return (unsigned int)v4;
        goto LABEL_9;
      }
    }
    else
    {
      LODWORD(v4) = 1;
      if ( !(_BYTE)v2 )
        return (unsigned int)v4;
      v8 = v10;
      LODWORD(v4) = v2;
      if ( !v10 )
        return (unsigned int)v4;
    }
    if ( (v8 & 4) == 0 )
      goto LABEL_8;
    goto LABEL_29;
  }
  if ( !*(_BYTE *)(v7 + 872) )
    goto LABEL_2;
LABEL_9:
  v5 = *(_QWORD *)(a1 + 72);
  LODWORD(v4) = 1;
  if ( v5 )
    LOBYTE(v4) = (unsigned int)*(unsigned __int8 *)(v5 + 8) - 15 > 1;
  return (unsigned int)v4;
}
