// Function: sub_37BD550
// Address: 0x37bd550
//
__int64 __fastcall sub_37BD550(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v3; // r11d
  int v6; // r11d
  __int64 v7; // r8
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  int v15; // r12d
  _QWORD *v16; // rbx

  v3 = *(_DWORD *)(a1 + 24);
  if ( v3 )
  {
    v6 = v3 - 1;
    v7 = *(_QWORD *)(a1 + 8);
    v8 = *a2;
    v9 = HIDWORD(*a2);
    v10 = 0x9DDFEA08EB382D69LL * (v9 ^ (((8 * *a2) & 0x7FFFFFFF8LL) + 12995744));
    v11 = v6
        & (-348639895
         * (((unsigned int)((0x9DDFEA08EB382D69LL * (v10 ^ v9 ^ (v10 >> 47))) >> 32) >> 15)
          ^ (-348639895 * (v10 ^ v9 ^ (v10 >> 47)))));
    v12 = (_QWORD *)(v7 + 16LL * v11);
    v13 = *v12;
    if ( v8 == *v12 )
    {
LABEL_3:
      *a3 = v12;
      return 1;
    }
    else
    {
      v15 = 1;
      v16 = 0;
      while ( unk_5051170 != v13 )
      {
        if ( qword_5051168 == v13 && !v16 )
          v16 = v12;
        v11 = v6 & (v15 + v11);
        v12 = (_QWORD *)(v7 + 16LL * v11);
        v13 = *v12;
        if ( *v12 == v8 )
          goto LABEL_3;
        ++v15;
      }
      if ( !v16 )
        v16 = v12;
      *a3 = v16;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
