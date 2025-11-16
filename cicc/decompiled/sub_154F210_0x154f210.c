// Function: sub_154F210
// Address: 0x154f210
//
__int64 __fastcall sub_154F210(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  unsigned int v6; // r8d
  __int64 v7; // rcx
  unsigned int v8; // esi
  unsigned int *v9; // rdx
  __int64 v10; // rdi
  int v12; // edx
  int v13; // r9d

  if ( *(_QWORD *)a1 )
  {
    sub_154EE80((_BYTE *)a1, a2, a3, a4);
    *(_QWORD *)a1 = 0;
  }
  if ( *(_QWORD *)(a1 + 8) && !*(_BYTE *)(a1 + 16) )
    sub_154F040(a1);
  v5 = *(unsigned int *)(a1 + 56);
  v6 = -1;
  if ( (_DWORD)v5 )
  {
    v7 = *(_QWORD *)(a1 + 40);
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (unsigned int *)(v7 + 16LL * (((_DWORD)v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
    v10 = *(_QWORD *)v9;
    if ( a2 == *(_QWORD *)v9 )
    {
LABEL_8:
      if ( v9 != (unsigned int *)(v7 + 16 * v5) )
        return v9[2];
    }
    else
    {
      v12 = 1;
      while ( v10 != -8 )
      {
        v13 = v12 + 1;
        v8 = (v5 - 1) & (v12 + v8);
        v9 = (unsigned int *)(v7 + 16LL * v8);
        v10 = *(_QWORD *)v9;
        if ( a2 == *(_QWORD *)v9 )
          goto LABEL_8;
        v12 = v13;
      }
    }
    return (unsigned int)-1;
  }
  return v6;
}
