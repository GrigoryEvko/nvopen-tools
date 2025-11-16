// Function: sub_E26E30
// Address: 0xe26e30
//
unsigned __int64 __fastcall sub_E26E30(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // r13
  __int64 *v4; // r15
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r14
  __int64 *v8; // r15
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // al
  __int64 v20; // rax
  _BYTE *v21; // rdx
  __int64 v22; // r14
  unsigned __int64 *v23; // rax
  unsigned __int64 *v24; // r15
  unsigned __int64 v25; // rdx

  v2 = *(_QWORD **)(a1 + 16);
  v3 = (*v2 + v2[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v2[1] = v3 - *v2 + 32;
  v4 = *(__int64 **)(a1 + 16);
  v5 = v4[1];
  if ( v5 > v4[2] )
  {
    v13 = (__int64 *)sub_22077B0(32);
    v4 = v13;
    if ( v13 )
    {
      *v13 = 0;
      v13[1] = 0;
      v13[2] = 0;
      v13[3] = 0;
    }
    v14 = sub_2207820(4096);
    v4[2] = 4096;
    *v4 = v14;
    v3 = v14;
    v15 = *(_QWORD *)(a1 + 16);
    v4[1] = 32;
    v4[3] = v15;
    *(_QWORD *)(a1 + 16) = v4;
    if ( !v3 )
    {
      v4[1] = 64;
      v7 = 32;
      goto LABEL_14;
    }
    *(_DWORD *)(v3 + 8) = 26;
    v6 = v3;
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)(v3 + 24) = 0;
    *(_QWORD *)v3 = &unk_49E11B8;
    v5 = 32;
  }
  else if ( v3 )
  {
    *(_DWORD *)(v3 + 8) = 26;
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)(v3 + 24) = 0;
    *(_QWORD *)v3 = &unk_49E11B8;
    v4 = *(__int64 **)(a1 + 16);
    v6 = *v4;
    v5 = v4[1];
  }
  else
  {
    v6 = *v4;
  }
  v7 = (v6 + v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v7 - v6 + 32;
  v8 = *(__int64 **)(a1 + 16);
  v9 = v8[1];
  if ( v9 <= v8[2] )
  {
    if ( !v7 )
    {
      v10 = *v8;
      goto LABEL_7;
    }
LABEL_14:
    *(_DWORD *)(v7 + 8) = 6;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)v7 = &unk_49E12B8;
    v8 = *(__int64 **)(a1 + 16);
    v10 = *v8;
    v9 = v8[1];
    goto LABEL_7;
  }
  v16 = (__int64 *)sub_22077B0(32);
  v8 = v16;
  if ( v16 )
  {
    *v16 = 0;
    v16[1] = 0;
    v16[2] = 0;
    v16[3] = 0;
  }
  v17 = sub_2207820(4096);
  v8[2] = 4096;
  *v8 = v17;
  v7 = v17;
  v18 = *(_QWORD *)(a1 + 16);
  v8[1] = 32;
  v8[3] = v18;
  *(_QWORD *)(a1 + 16) = v8;
  if ( !v7 )
  {
    v8[1] = 112;
    v11 = 32;
    goto LABEL_9;
  }
  *(_DWORD *)(v7 + 8) = 6;
  v10 = v7;
  *(_QWORD *)(v7 + 16) = 0;
  *(_QWORD *)(v7 + 24) = 0;
  *(_QWORD *)v7 = &unk_49E12B8;
  v9 = 32;
LABEL_7:
  v11 = (v10 + v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v8[1] = v11 - v10 + 80;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v23 = (unsigned __int64 *)sub_22077B0(32);
    v24 = v23;
    if ( v23 )
    {
      *v23 = 0;
      v23[1] = 0;
      v23[2] = 0;
      v23[3] = 0;
    }
    v11 = sub_2207820(4096);
    v25 = *(_QWORD *)(a1 + 16);
    v24[2] = 4096;
    *v24 = v11;
    v24[3] = v25;
    *(_QWORD *)(a1 + 16) = v24;
    v24[1] = 80;
  }
  if ( !v11 )
  {
    *(_QWORD *)(v3 + 24) = 0;
    MEMORY[0x16] = 0;
    BUG();
  }
LABEL_9:
  *(_BYTE *)(v11 + 12) = 0;
  *(_WORD *)(v11 + 22) = 8;
  *(_DWORD *)(v11 + 8) = 13;
  *(_DWORD *)(v11 + 16) = 0;
  *(_QWORD *)v11 = &unk_49E10B0;
  *(_BYTE *)(v11 + 20) = 0;
  *(_DWORD *)(v11 + 24) = 0;
  *(_QWORD *)(v11 + 32) = 0;
  *(_BYTE *)(v11 + 40) = 0;
  *(_QWORD *)(v11 + 48) = 0;
  *(_BYTE *)(v11 + 56) = 0;
  *(_QWORD *)(v11 + 60) = 0;
  *(_QWORD *)(v11 + 68) = 0;
  *(_QWORD *)(v3 + 24) = v11;
  *(_WORD *)(v11 + 22) = 256;
  *(_QWORD *)(v3 + 16) = sub_E263F0(a1, a2, v7);
  if ( !*(_BYTE *)(a1 + 8) )
  {
    v19 = sub_E20730((size_t *)a2, 2u, "$B") ^ 1;
    *(_BYTE *)(a1 + 8) = v19;
    if ( !v19 )
    {
      *(_QWORD *)(v7 + 24) = sub_E21AA0(a1, (unsigned __int64 *)a2);
      if ( !*(_BYTE *)(a1 + 8) )
      {
        v20 = *a2;
        if ( *a2 && (v21 = (_BYTE *)a2[1], *v21 == 65) )
        {
          a2[1] = (__int64)(v21 + 1);
          *a2 = v20 - 1;
          *(_BYTE *)(a1 + 8) = 0;
          v22 = *(_QWORD *)(v3 + 24);
          *(_BYTE *)(v22 + 20) = sub_E22DC0(a1, a2);
          if ( !*(_BYTE *)(a1 + 8) )
            return v3;
        }
        else
        {
          *(_BYTE *)(a1 + 8) = 1;
        }
      }
    }
  }
  return 0;
}
