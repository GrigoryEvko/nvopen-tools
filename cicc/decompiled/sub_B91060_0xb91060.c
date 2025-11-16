// Function: sub_B91060
// Address: 0xb91060
//
unsigned __int64 __fastcall sub_B91060(unsigned __int8 *a1)
{
  int v1; // eax
  __int64 v2; // r12
  unsigned __int64 *v4; // rax
  unsigned __int64 *v5; // r13
  _QWORD *v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r12
  __int64 v11; // r13

  v1 = *a1;
  if ( (unsigned __int8)(v1 - 5) > 0x1Fu )
  {
    if ( (_BYTE)v1 == 4 || (unsigned int)(v1 - 1) <= 1 )
      return (unsigned __int64)(a1 + 8);
    return 0;
  }
  if ( (a1[1] & 0x7F) != 2 && !*((_DWORD *)a1 - 2) && (_BYTE)v1 != 30 )
    return 0;
  v2 = *((_QWORD *)a1 + 1);
  if ( (v2 & 4) != 0 )
    return v2 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = (unsigned __int64 *)sub_22077B0(128);
  v5 = v4;
  if ( v4 )
  {
    v4[1] = 0;
    v6 = v4 + 16;
    v4[2] = 0;
    v4[3] = 1;
    *v4 = v2 & 0xFFFFFFFFFFFFFFF8LL;
    v7 = v4 + 4;
    do
    {
      if ( v7 )
        *v7 = -4096;
      v7 += 3;
    }
    while ( v7 != v6 );
    v8 = *((_QWORD *)a1 + 1);
    if ( (v8 & 4) == 0 )
      goto LABEL_21;
  }
  else
  {
    v8 = *((_QWORD *)a1 + 1);
    if ( (v8 & 4) == 0 )
    {
      *((_QWORD *)a1 + 1) = 4;
      v2 = 4;
      return v2 & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = v9;
  if ( v9 )
  {
    if ( (*(_BYTE *)(v9 + 24) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v9 + 32), 24LL * *(unsigned int *)(v9 + 40), 8);
    j_j___libc_free_0(v10, 128);
  }
LABEL_21:
  v11 = (unsigned __int64)v5 | 4;
  *((_QWORD *)a1 + 1) = v11;
  v2 = v11;
  if ( (v11 & 4) != 0 )
    return v2 & 0xFFFFFFFFFFFFFFF8LL;
  return 0;
}
