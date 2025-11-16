// Function: sub_3158140
// Address: 0x3158140
//
bool __fastcall sub_3158140(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // edx
  __int64 *v5; // rcx
  __int64 v6; // r9
  bool result; // al
  __int64 v8; // rax
  __int64 v9; // r8
  unsigned int v10; // edx
  __int64 *v11; // rcx
  __int64 v12; // rdi
  int v13; // ecx
  int v14; // ecx
  int v15; // r11d
  int v16; // r10d

  v2 = *(unsigned int *)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 24);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 88LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 88 * v2) )
      {
        result = 0;
        if ( *((_DWORD *)v5 + 12) )
          return result;
      }
    }
    else
    {
      v14 = 1;
      while ( v6 != -4096 )
      {
        v15 = v14 + 1;
        v4 = (v2 - 1) & (v14 + v4);
        v5 = (__int64 *)(v3 + 88LL * v4);
        v6 = *v5;
        if ( a2 == *v5 )
          goto LABEL_3;
        v14 = v15;
      }
    }
  }
  v8 = *(unsigned int *)(a1 + 72);
  v9 = *(_QWORD *)(a1 + 56);
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (__int64 *)(v9 + 88LL * v10);
    v12 = *v11;
    if ( a2 == *v11 )
    {
LABEL_7:
      if ( v11 != (__int64 *)(v9 + 88 * v8) )
        return *((_DWORD *)v11 + 12) == 0;
    }
    else
    {
      v13 = 1;
      while ( v12 != -4096 )
      {
        v16 = v13 + 1;
        v10 = (v8 - 1) & (v13 + v10);
        v11 = (__int64 *)(v9 + 88LL * v10);
        v12 = *v11;
        if ( a2 == *v11 )
          goto LABEL_7;
        v13 = v16;
      }
    }
  }
  return 1;
}
