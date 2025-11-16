// Function: sub_2398550
// Address: 0x2398550
//
void __fastcall sub_2398550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned int v7; // eax
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // rax
  __int64 *v15; // r12
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // [rsp-48h] [rbp-48h]
  __int64 v26[8]; // [rsp-40h] [rbp-40h] BYREF

  LODWORD(v6) = *(_DWORD *)(a1 + 8) & 0xFFFFFFFE;
  v7 = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE;
  *(_DWORD *)(a2 + 8) = v6 | *(_DWORD *)(a2 + 8) & 1;
  *(_DWORD *)(a1 + 8) = v7 | *(_DWORD *)(a1 + 8) & 1;
  v8 = *(_DWORD *)(a1 + 12);
  LODWORD(v9) = *(_DWORD *)(a2 + 12);
  *(_DWORD *)(a1 + 12) = v9;
  *(_DWORD *)(a2 + 12) = v8;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
      goto LABEL_4;
    v15 = (__int64 *)(a1 + 24);
    v16 = (__int64 *)(a2 + 16);
    while ( 1 )
    {
      v17 = *(v15 - 1);
      LOBYTE(v9) = v17 != -4096;
      LOBYTE(v6) = v17 != -8192;
      v9 = (unsigned int)v6 & (unsigned int)v9;
      v6 = *v16;
      if ( *v16 == -4096 )
      {
        *(v15 - 1) = -4096;
        *v16 = v17;
        if ( (_BYTE)v9 )
          goto LABEL_21;
      }
      else
      {
        if ( v6 != -8192 )
        {
          if ( (_BYTE)v9 )
          {
            v25 = *(v15 - 1);
            v20 = *v15;
            *v15 = 0;
            v26[0] = v20;
            *(v15 - 1) = *v16;
            sub_2361BD0(v15, v16 + 1, v9, v6, a5, a6);
            *v16 = v25;
            sub_2361BD0(v16 + 1, v26, v21, v22, v23, v24);
            sub_2396550(v26);
          }
          else
          {
            *(v15 - 1) = v6;
            *v16 = v17;
            *v15 = v16[1];
            v16[1] = 0;
            sub_2396550(v16 + 1);
          }
          goto LABEL_17;
        }
        *(v15 - 1) = -8192;
        *v16 = v17;
        if ( (_BYTE)v9 )
        {
LABEL_21:
          v16[1] = *v15;
          *v15 = 0;
          sub_2396550(v15);
        }
      }
LABEL_17:
      v16 += 2;
      v15 += 2;
      if ( (__int64 *)(a2 + 48) == v16 )
        return;
    }
  }
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v18 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    v19 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v18;
    LODWORD(v18) = *(_DWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v19;
    *(_DWORD *)(a2 + 24) = v18;
    return;
  }
  v10 = a2;
  a2 = a1;
  a1 = v10;
LABEL_4:
  *(_BYTE *)(a2 + 8) |= 1u;
  v11 = *(_QWORD *)(a1 + 16);
  v12 = *(_QWORD *)(a2 + 16);
  v13 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)(a2 + 16) = v11;
  if ( v11 != -4096 && v11 != -8192 )
    *(_QWORD *)(a2 + 24) = *(_QWORD *)(a1 + 24);
  v14 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a2 + 32) = v14;
  if ( v14 != -8192 && v14 != -4096 )
    *(_QWORD *)(a2 + 40) = *(_QWORD *)(a1 + 40);
  *(_BYTE *)(a1 + 8) &= ~1u;
  *(_QWORD *)(a1 + 16) = v12;
  *(_DWORD *)(a1 + 24) = v13;
}
