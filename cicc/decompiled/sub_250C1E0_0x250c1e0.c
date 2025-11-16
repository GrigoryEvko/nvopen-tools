// Function: sub_250C1E0
// Address: 0x250c1e0
//
char __fastcall sub_250C1E0(unsigned __int8 **a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // al
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rax
  _BYTE *v12; // rax
  _BYTE *j; // r12
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r10d
  unsigned int i; // eax
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rdi

  v2 = *a1;
  v3 = **a1;
  if ( v3 <= 0x15u )
    return 1;
  v4 = (__int64)a1[1];
  if ( (unsigned __int8 *)v4 == v2 )
    return 1;
  v5 = 0;
  if ( v4 )
  {
    v6 = sub_B43CB0((__int64)a1[1]);
    v2 = *a1;
    v5 = v6;
    v3 = **a1;
  }
  if ( v3 == 22 )
    return *((_QWORD *)v2 + 3) == v5;
  if ( v3 <= 0x1Cu || v5 != sub_B43CB0((__int64)v2) )
    return 0;
  v8 = *(_QWORD *)(a2 + 240);
  v9 = *(_QWORD *)v8;
  if ( *(_QWORD *)v8 )
  {
    if ( !*(_BYTE *)(v8 + 16) )
    {
      v10 = sub_BC1CD0(v9, &unk_4F81450, v5) + 8;
      return sub_B19DB0(v10, (__int64)v2, v4);
    }
    v14 = *(unsigned int *)(v9 + 88);
    v15 = *(_QWORD *)(v9 + 72);
    if ( (_DWORD)v14 )
    {
      v16 = 1;
      for ( i = (v14 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = (v14 - 1) & v19 )
      {
        v18 = v15 + 24LL * i;
        if ( *(_UNKNOWN **)v18 == &unk_4F81450 && v5 == *(_QWORD *)(v18 + 8) )
          break;
        if ( *(_QWORD *)v18 == -4096 && *(_QWORD *)(v18 + 8) == -4096 )
          goto LABEL_16;
        v19 = v16 + i;
        ++v16;
      }
      if ( v18 != v15 + 24 * v14 )
      {
        v20 = *(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL);
        if ( v20 )
        {
          v10 = v20 + 8;
          return sub_B19DB0(v10, (__int64)v2, v4);
        }
      }
    }
  }
LABEL_16:
  if ( !v4 )
    return 0;
  v11 = *((_QWORD *)v2 + 5);
  if ( *(_QWORD *)(v4 + 40) != v11 )
    return 0;
  v12 = (_BYTE *)(v11 + 48);
  for ( j = v2 + 24; v12 != j; j = (_BYTE *)*((_QWORD *)j + 1) )
  {
    if ( j && (_BYTE *)v4 == j - 24 )
      break;
  }
  return v12 != j;
}
