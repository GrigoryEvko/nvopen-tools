// Function: sub_20DFA10
// Address: 0x20dfa10
//
unsigned __int64 __fastcall sub_20DFA10(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 *v4; // r13
  unsigned __int64 v5; // r12
  int v6; // r9d
  __int64 v7; // rdx
  int v8; // r8d
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r14
  _QWORD *v14; // rsi
  char *v15; // r13
  unsigned __int64 v16; // r9
  char *v17; // rcx
  __int64 v18; // rax
  _QWORD v20[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_1E0B6F0(*(_QWORD *)(a1 + 456), *(_QWORD *)(a2 + 40));
  v4 = *(__int64 **)(a2 + 8);
  v5 = (unsigned __int64)v3;
  sub_1DD8DC0(*(_QWORD *)(a1 + 456) + 320LL, (__int64)v3);
  v7 = *v4;
  v12 = *(_QWORD *)v5;
  *(_QWORD *)(v5 + 8) = v4;
  v20[0] = 0;
  v8 = a1 + 232;
  v7 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v7 | v12 & 7;
  *(_QWORD *)(v7 + 8) = v5;
  *v4 = v5 | *v4 & 7;
  v9 = *(unsigned int *)(a1 + 240);
  v10 = *(_QWORD *)(a1 + 232);
  v11 = 8 * v9;
  LODWORD(v12) = *(_DWORD *)(a1 + 240);
  v13 = 8LL * *(int *)(v5 + 48);
  v14 = (_QWORD *)(v10 + 8 * v9);
  v15 = (char *)(v10 + v13);
  if ( (_QWORD *)(v10 + v13) == v14 )
  {
    if ( (unsigned int)v9 >= *(_DWORD *)(a1 + 244) )
    {
      sub_16CD150(a1 + 232, (const void *)(a1 + 248), 0, 8, v8, v6);
      v14 = (_QWORD *)(*(_QWORD *)(a1 + 232) + 8LL * *(unsigned int *)(a1 + 240));
    }
    *v14 = 0;
    ++*(_DWORD *)(a1 + 240);
    return v5;
  }
  else
  {
    v16 = *(unsigned int *)(a1 + 244);
    if ( v9 >= v16 )
    {
      sub_16CD150(a1 + 232, (const void *)(a1 + 248), 0, 8, v8, v16);
      v10 = *(_QWORD *)(a1 + 232);
      v12 = *(unsigned int *)(a1 + 240);
      v11 = 8 * v12;
      v15 = (char *)(v10 + v13);
      v14 = (_QWORD *)(v10 + 8 * v12);
    }
    v17 = (char *)(v10 + v11 - 8);
    if ( v14 )
    {
      *v14 = *(_QWORD *)v17;
      v10 = *(_QWORD *)(a1 + 232);
      v12 = *(unsigned int *)(a1 + 240);
      v11 = 8 * v12;
      v17 = (char *)(v10 + 8 * v12 - 8);
    }
    if ( v17 != v15 )
    {
      memmove((void *)(v10 + v11 - (v17 - v15)), v15, v17 - v15);
      LODWORD(v12) = *(_DWORD *)(a1 + 240);
    }
    v18 = (unsigned int)(v12 + 1);
    *(_DWORD *)(a1 + 240) = v18;
    if ( v15 > (char *)v20 || (unsigned __int64)v20 >= *(_QWORD *)(a1 + 232) + 8 * v18 )
    {
      *(_QWORD *)v15 = v20[0];
      return v5;
    }
    else
    {
      *(_QWORD *)v15 = v20[1];
      return v5;
    }
  }
}
