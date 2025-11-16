// Function: sub_E208B0
// Address: 0xe208b0
//
unsigned __int64 __fastcall sub_E208B0(__int64 **a1, _QWORD *a2, __int64 a3)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // r12
  size_t v8; // rdx
  __int64 v9; // rdi
  char *v10; // rcx
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // r13
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // [rsp+8h] [rbp-38h]

  v6 = **a1;
  v7 = (v6 + (*a1)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a1)[1] = v7 - v6 + 32;
  if ( (*a1)[1] > (unsigned __int64)(*a1)[2] )
  {
    v18 = (__int64 *)sub_22077B0(32);
    v19 = v18;
    if ( v18 )
    {
      *v18 = 0;
      v18[1] = 0;
      v18[2] = 0;
      v18[3] = 0;
    }
    v20 = sub_2207820(4096);
    v19[2] = 4096;
    *v19 = v20;
    v7 = v20;
    v21 = *a1;
    v19[1] = 32;
    v19[3] = (__int64)v21;
    *a1 = v19;
  }
  if ( !v7 )
  {
    MEMORY[0x18] = 0;
    BUG();
  }
  *(_QWORD *)(v7 + 24) = 0;
  v8 = 8 * a3;
  *(_DWORD *)(v7 + 8) = 19;
  *(_QWORD *)(v7 + 24) = a3;
  *(_QWORD *)v7 = &unk_49E0EC0;
  *(_QWORD *)(v7 + 16) = 0;
  v9 = **a1;
  v10 = (char *)((v9 + (*a1)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL);
  (*a1)[1] = (__int64)&v10[8 * a3 - v9];
  if ( (*a1)[1] > (unsigned __int64)(*a1)[2] )
  {
    v13 = 4096;
    if ( v8 >= 0x1000 )
      v13 = 8 * a3;
    v22 = v13;
    v14 = (__int64 *)sub_22077B0(32);
    v15 = v14;
    if ( v14 )
    {
      *v14 = 0;
      v14[1] = 0;
      v14[2] = 0;
      v14[3] = 0;
    }
    v16 = sub_2207820(v22);
    v8 = 8 * a3;
    *v15 = v16;
    v10 = (char *)v16;
    v17 = *a1;
    v15[1] = 8 * a3;
    v15[3] = (__int64)v17;
    *a1 = v15;
    v15[2] = v22;
    if ( v10 )
      goto LABEL_5;
LABEL_19:
    v10 = 0;
    goto LABEL_9;
  }
  if ( !v10 )
    goto LABEL_19;
LABEL_5:
  if ( a3 - 1 >= 0 )
  {
    if ( a3 - 2 < -1 )
      v8 = 8;
    v10 = (char *)memset(v10, 0, v8);
  }
LABEL_9:
  *(_QWORD *)(v7 + 16) = v10;
  if ( a3 )
  {
    v11 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v10[8 * v11++] = *a2;
      a2 = (_QWORD *)a2[1];
      if ( a3 == v11 )
        break;
      v10 = *(char **)(v7 + 16);
    }
  }
  return v7;
}
