// Function: sub_B738D0
// Address: 0xb738d0
//
void __fastcall sub_B738D0(__int64 a1)
{
  __int64 v1; // r12
  int *v2; // rbx
  int v3; // eax
  int *v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rdi

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(int **)(a1 + 8);
    v3 = *v2;
    v4 = &v2[8 * v1];
    if ( *v2 == -1 )
      goto LABEL_16;
LABEL_4:
    if ( v3 == -2 && !*((_BYTE *)v2 + 4) && !v2[4] && *((_QWORD *)v2 + 1) == -2 )
      goto LABEL_14;
LABEL_6:
    v5 = *((_QWORD *)v2 + 3);
    if ( v5 )
    {
      if ( *(_DWORD *)(v5 + 32) > 0x40u )
      {
        v6 = *(_QWORD *)(v5 + 24);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      sub_BD7260(v5);
      sub_BD2DD0(v5);
    }
    if ( (unsigned int)v2[4] > 0x40 )
    {
      v7 = *((_QWORD *)v2 + 1);
      if ( v7 )
        j_j___libc_free_0_0(v7);
    }
LABEL_14:
    while ( 1 )
    {
      v2 += 8;
      if ( v4 == v2 )
        break;
      v3 = *v2;
      if ( *v2 != -1 )
        goto LABEL_4;
LABEL_16:
      if ( !*((_BYTE *)v2 + 4) || v2[4] || *((_QWORD *)v2 + 1) != -1 )
        goto LABEL_6;
    }
  }
}
