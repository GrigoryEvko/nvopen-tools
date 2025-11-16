// Function: sub_ACD780
// Address: 0xacd780
//
void __fastcall sub_ACD780(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  _DWORD *v12; // r15
  bool v13; // cc
  __int64 v14; // rdi
  __int64 v15; // rdx
  _DWORD *v16; // rax
  __int64 v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 *v20; // [rsp+8h] [rbp-58h]
  _DWORD *v21; // [rsp+10h] [rbp-50h] BYREF
  __int64 v22; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v23; // [rsp+20h] [rbp-40h]

  v5 = a2;
  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  BYTE4(v21) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  LODWORD(v21) = -1;
  v8 = v7 + 32 * v6;
  v23 = 0;
  v22 = -1;
  if ( v7 == v8 )
    goto LABEL_11;
  v9 = &v22;
  do
  {
    while ( 1 )
    {
      if ( !v7 )
        goto LABEL_4;
      *(_QWORD *)v7 = v21;
      v10 = v23;
      *(_DWORD *)(v7 + 16) = v23;
      if ( v10 > 0x40 )
        break;
      *(_QWORD *)(v7 + 8) = v22;
LABEL_4:
      v7 += 32;
      if ( v8 == v7 )
        goto LABEL_8;
    }
    v11 = v7 + 8;
    v7 += 32;
    v20 = v9;
    sub_C43780(v11, v9);
    v9 = v20;
  }
  while ( v8 != v7 );
LABEL_8:
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
LABEL_11:
  if ( a2 != a3 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v5 == -1 )
      {
        if ( !*(_BYTE *)(v5 + 4) || *(_DWORD *)(v5 + 16) || *(_QWORD *)(v5 + 8) != -1 )
        {
LABEL_15:
          sub_AC6600(a1, (int *)v5, &v21);
          v12 = v21;
          *v21 = *(_DWORD *)v5;
          v13 = v12[4] <= 0x40u;
          *((_BYTE *)v12 + 4) = *(_BYTE *)(v5 + 4);
          if ( !v13 )
          {
            v14 = *((_QWORD *)v12 + 1);
            if ( v14 )
              j_j___libc_free_0_0(v14);
          }
          *((_QWORD *)v12 + 1) = *(_QWORD *)(v5 + 8);
          v12[4] = *(_DWORD *)(v5 + 16);
          v15 = *(_QWORD *)(v5 + 24);
          v16 = v21;
          *(_DWORD *)(v5 + 16) = 0;
          *((_QWORD *)v16 + 3) = v15;
          *(_QWORD *)(v5 + 24) = 0;
          ++*(_DWORD *)(a1 + 16);
          v17 = *(_QWORD *)(v5 + 24);
          if ( v17 )
          {
            if ( *(_DWORD *)(v17 + 32) > 0x40u )
            {
              v19 = *(_QWORD *)(v17 + 24);
              if ( v19 )
                j_j___libc_free_0_0(v19);
            }
            sub_BD7260(v17);
            sub_BD2DD0(v17);
          }
          if ( *(_DWORD *)(v5 + 16) > 0x40u )
          {
            v18 = *(_QWORD *)(v5 + 8);
            if ( v18 )
              j_j___libc_free_0_0(v18);
          }
        }
      }
      else if ( *(_DWORD *)v5 != -2 || *(_BYTE *)(v5 + 4) || *(_DWORD *)(v5 + 16) || *(_QWORD *)(v5 + 8) != -2 )
      {
        goto LABEL_15;
      }
      v5 += 32;
      if ( a3 == v5 )
        return;
      continue;
    }
  }
}
