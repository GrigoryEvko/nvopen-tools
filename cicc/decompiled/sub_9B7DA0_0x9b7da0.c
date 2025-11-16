// Function: sub_9B7DA0
// Address: 0x9b7da0
//
bool __fastcall sub_9B7DA0(char *a1, unsigned int a2, unsigned int a3)
{
  unsigned __int8 v5; // dl
  bool result; // al
  int *v7; // rax
  int v8; // eax
  unsigned __int8 v9; // al
  __int64 v10; // rdi
  char *v11; // r14
  __int64 v12; // r13
  __int64 v13; // rdx
  size_t v14; // rdx
  int v15; // r8d
  __int64 *v16; // r12
  __int64 v17; // rdi
  __int64 v18; // r15

  while ( 1 )
  {
    v5 = *a1;
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a1 + 1) + 8LL) - 17 <= 1 )
    {
      if ( (unsigned int)v5 - 12 <= 1 )
        return 1;
      if ( v5 <= 0x15u )
        return sub_AD7630(a1, 0) != 0;
    }
    if ( v5 == 92 )
      break;
    v7 = (int *)sub_C94E20(qword_4F862D0);
    if ( v7 )
      v8 = *v7;
    else
      v8 = qword_4F862D0[2];
    if ( a3 == v8 )
      return 0;
    v9 = *a1;
    if ( (unsigned __int8)*a1 <= 0x1Cu )
      return 0;
    ++a3;
    if ( (unsigned int)v9 - 42 <= 0x11 && (v10 = *((_QWORD *)a1 - 8)) != 0 && (v11 = (char *)*((_QWORD *)a1 - 4)) != 0 )
    {
      if ( !(unsigned __int8)sub_9B7DA0(v10, a2, a3) )
        return 0;
    }
    else
    {
      if ( v9 != 86 )
        return 0;
      if ( (a1[7] & 0x40) != 0 )
      {
        v16 = (__int64 *)*((_QWORD *)a1 - 1);
        v17 = *v16;
        if ( !*v16 )
          return 0;
      }
      else
      {
        v16 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v17 = *v16;
        if ( !*v16 )
          return 0;
      }
      v18 = v16[4];
      if ( !v18 )
        return 0;
      v11 = (char *)v16[8];
      if ( !v11 || !(unsigned __int8)sub_9B7DA0(v17, a2, a3) || !(unsigned __int8)sub_9B7DA0(v18, a2, a3) )
        return 0;
    }
    a1 = v11;
  }
  v12 = *((_QWORD *)a1 + 9);
  v13 = 4LL * *((unsigned int *)a1 + 20);
  if ( !v13
    || (v14 = v13 - 4) == 0
    || (v15 = memcmp((const void *)(v12 + 4), *((const void **)a1 + 9), v14), result = 0, !v15) )
  {
    if ( a2 == -1 )
      return 1;
    return *(_DWORD *)(v12 + 4LL * a2) == a2;
  }
  return result;
}
