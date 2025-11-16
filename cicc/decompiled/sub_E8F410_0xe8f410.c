// Function: sub_E8F410
// Address: 0xe8f410
//
__int64 __fastcall sub_E8F410(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r14
  unsigned int v3; // r13d
  _QWORD *v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // r15
  _QWORD *v7; // rax
  __int64 v9; // [rsp+8h] [rbp-38h]

  v1 = a1;
  v2 = *a1;
  v9 = a1[1];
  while ( 1 )
  {
    v4 = *(_QWORD **)v2;
    v5 = *(v1 - 2);
    v6 = v1;
    if ( !*(_QWORD *)v2 )
    {
      if ( (*(_BYTE *)(v2 + 9) & 0x70) != 0x20 || *(char *)(v2 + 8) < 0 )
        BUG();
      *(_BYTE *)(v2 + 8) |= 8u;
      v4 = sub_E807D0(*(_QWORD *)(v2 + 24));
      *(_QWORD *)v2 = v4;
    }
    v3 = *(_DWORD *)(v4[1] + 36LL);
    if ( !*(_QWORD *)v5 )
      break;
    v1 -= 2;
    if ( v3 >= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v5 + 8LL) + 36LL) )
      goto LABEL_12;
LABEL_4:
    v1[2] = *v1;
    v1[3] = v1[1];
  }
  if ( (*(_BYTE *)(v5 + 9) & 0x70) != 0x20 || *(char *)(v5 + 8) < 0 )
    BUG();
  *(_BYTE *)(v5 + 8) |= 8u;
  v1 -= 2;
  v7 = sub_E807D0(*(_QWORD *)(v5 + 24));
  *(_QWORD *)v5 = v7;
  if ( v3 < *(_DWORD *)(v7[1] + 36LL) )
    goto LABEL_4;
LABEL_12:
  *v6 = v2;
  v6[1] = v9;
  return v9;
}
