// Function: sub_E21710
// Address: 0xe21710
//
unsigned __int64 __fastcall sub_E21710(__int64 a1, size_t *a2)
{
  size_t v2; // rbx
  size_t v3; // r12
  char *v4; // rax
  char *v5; // rax
  char *v6; // rax
  _QWORD *v7; // rax
  size_t v8; // r13
  unsigned __int64 v9; // r14
  __int64 *v10; // rcx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 *v20; // [rsp+8h] [rbp-38h]
  __int64 *v21; // [rsp+8h] [rbp-38h]

  v2 = *a2;
  if ( *a2 <= 3
    || (v3 = a2[1], (v4 = (char *)memchr((const void *)(v3 + 3), 64, v2 - 3)) == 0)
    || (v5 = &v4[-v3], v5 == (char *)-1LL) )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  v6 = v5 + 1;
  a2[1] = (size_t)&v6[v3];
  *a2 = v2 - (_QWORD)v6;
  sub_E20730(a2, 6u, "??_R4@");
  v7 = *(_QWORD **)(a1 + 16);
  v8 = *a2;
  v9 = (*v7 + v7[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v7[1] = v9 - *v7 + 24;
  v10 = *(__int64 **)(a1 + 16);
  v11 = v10[1];
  if ( v11 <= v10[2] )
  {
    if ( v9 )
    {
      *(_DWORD *)(v9 + 8) = 1;
      *(_QWORD *)(v9 + 16) = 0;
      *(_QWORD *)v9 = &unk_49E1190;
      v10 = *(__int64 **)(a1 + 16);
      v12 = *v10;
      v11 = v10[1];
    }
    else
    {
      v12 = *v10;
    }
    goto LABEL_7;
  }
  v17 = (__int64 *)sub_22077B0(32);
  if ( v17 )
  {
    *v17 = 0;
    v17[1] = 0;
    v17[2] = 0;
    v17[3] = 0;
  }
  v21 = v17;
  v18 = sub_2207820(4096);
  v10 = v21;
  v9 = v18;
  *v21 = v18;
  v19 = *(_QWORD *)(a1 + 16);
  v21[2] = 4096;
  v21[3] = v19;
  *(_QWORD *)(a1 + 16) = v21;
  v21[1] = 24;
  if ( v9 )
  {
    *(_DWORD *)(v9 + 8) = 1;
    v12 = v9;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)v9 = &unk_49E1190;
    v11 = 24;
LABEL_7:
    v13 = (v12 + v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v10[1] = v13 - v12 + 40;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v15 = (unsigned __int64 *)sub_22077B0(32);
      if ( v15 )
      {
        *v15 = 0;
        v15[1] = 0;
        v15[2] = 0;
        v15[3] = 0;
      }
      v20 = v15;
      v13 = sub_2207820(4096);
      *v20 = v13;
      v16 = *(_QWORD *)(a1 + 16);
      v20[2] = 4096;
      v20[3] = v16;
      *(_QWORD *)(a1 + 16) = v20;
      v20[1] = 40;
    }
    if ( !v13 )
    {
      MEMORY[0x18] = 0;
      BUG();
    }
    goto LABEL_9;
  }
  v21[1] = 64;
  v13 = 24;
LABEL_9:
  *(_QWORD *)(v13 + 24) = 0;
  *(_QWORD *)(v13 + 32) = 0;
  *(_DWORD *)(v13 + 8) = 5;
  *(_QWORD *)(v13 + 16) = 0;
  *(_QWORD *)v13 = &unk_49E0F88;
  *(_QWORD *)(v13 + 24) = v2 - v8;
  *(_QWORD *)(v13 + 32) = v3;
  *(_QWORD *)(v9 + 16) = sub_E20AE0((__int64 **)(a1 + 16), v13);
  return v9;
}
