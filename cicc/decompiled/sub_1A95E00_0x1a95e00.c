// Function: sub_1A95E00
// Address: 0x1a95e00
//
unsigned __int64 __fastcall sub_1A95E00(__int64 a1)
{
  unsigned __int64 v1; // rbx
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned int *v10; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 v18; // rax

  v1 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(_BYTE *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  if ( (a1 & 4) != 0 )
  {
    if ( v2 < 0 )
    {
      v3 = sub_1648A40(v1);
      v5 = v3 + v4;
      if ( *(char *)(v1 + 23) < 0 )
        v5 -= sub_1648A40(v1);
      v6 = v5 >> 4;
      if ( (_DWORD)v6 )
      {
        v7 = 0;
        v8 = 16LL * (unsigned int)v6;
        while ( 1 )
        {
          v9 = 0;
          if ( *(char *)(v1 + 23) < 0 )
            v9 = sub_1648A40(v1);
          v10 = (unsigned int *)(v7 + v9);
          if ( !*(_DWORD *)(*(_QWORD *)v10 + 8LL) )
            break;
          v7 += 16;
          if ( v7 == v8 )
            return 0;
        }
        return 24LL * v10[2] - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) + v1;
      }
    }
  }
  else if ( v2 < 0 )
  {
    v12 = sub_1648A40(v1);
    v14 = v12 + v13;
    if ( *(char *)(v1 + 23) < 0 )
      v14 -= sub_1648A40(v1);
    v15 = v14 >> 4;
    if ( (_DWORD)v15 )
    {
      v16 = 0;
      v17 = 16LL * (unsigned int)v15;
      while ( 1 )
      {
        v18 = 0;
        if ( *(char *)(v1 + 23) < 0 )
          v18 = sub_1648A40(v1);
        v10 = (unsigned int *)(v16 + v18);
        if ( !*(_DWORD *)(*(_QWORD *)v10 + 8LL) )
          break;
        v16 += 16;
        if ( v17 == v16 )
          return 0;
      }
      return 24LL * v10[2] - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) + v1;
    }
  }
  return 0;
}
