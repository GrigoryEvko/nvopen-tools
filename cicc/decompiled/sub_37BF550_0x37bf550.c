// Function: sub_37BF550
// Address: 0x37bf550
//
__int64 __fastcall sub_37BF550(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v5; // r8
  int v6; // r9d
  __int64 v7; // rdi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  int v14; // edx
  int v15; // r12d
  _QWORD *v16; // rbx

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    v14 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    v6 = v14 - 1;
    if ( !v14 )
    {
      *a3 = 0;
      return 0;
    }
  }
  v7 = *a2;
  v8 = HIDWORD(*a2);
  v9 = 0x9DDFEA08EB382D69LL * (v8 ^ (((8 * *a2) & 0x7FFFFFFF8LL) + 12995744));
  v10 = v6
      & (-348639895
       * (((unsigned int)((0x9DDFEA08EB382D69LL * (v9 ^ v8 ^ (v9 >> 47))) >> 32) >> 15)
        ^ (-348639895 * (v9 ^ v8 ^ (v9 >> 47)))));
  v11 = (_QWORD *)(v5 + 16LL * v10);
  v12 = *v11;
  if ( v7 == *v11 )
  {
LABEL_4:
    *a3 = v11;
    return 1;
  }
  else
  {
    v15 = 1;
    v16 = 0;
    while ( unk_5051170 != v12 )
    {
      if ( qword_5051168 == v12 && !v16 )
        v16 = v11;
      v10 = v6 & (v15 + v10);
      v11 = (_QWORD *)(v5 + 16LL * v10);
      v12 = *v11;
      if ( v7 == *v11 )
        goto LABEL_4;
      ++v15;
    }
    if ( !v16 )
      v16 = v11;
    *a3 = v16;
    return 0;
  }
}
