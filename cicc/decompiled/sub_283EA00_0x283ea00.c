// Function: sub_283EA00
// Address: 0x283ea00
//
__int64 __fastcall sub_283EA00(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 **v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  void **v10; // rax
  __int64 **v12; // rsi

  if ( *a3 || *(_QWORD *)(a2 + 104) == *(_QWORD *)(a2 + 96) )
  {
    sub_283ED10();
    if ( *(_DWORD *)(a1 + 68) != *(_DWORD *)(a1 + 72) )
      goto LABEL_3;
  }
  else
  {
    sub_283DEB0(a1, a2, a3, a4, a5, a6);
    if ( *(_DWORD *)(a1 + 68) != *(_DWORD *)(a1 + 72) )
      goto LABEL_3;
  }
  if ( *(_BYTE *)(a1 + 28) )
  {
    v10 = *(void ***)(a1 + 8);
    v12 = (__int64 **)&v10[*(unsigned int *)(a1 + 20)];
    v7 = *(unsigned int *)(a1 + 20);
    v6 = (__int64 **)v10;
    if ( v10 != (void **)v12 )
    {
      while ( *v6 != &qword_4F82400 )
      {
        if ( v12 == ++v6 )
        {
LABEL_7:
          while ( *v10 != &unk_4FDBCE8 )
          {
            if ( v6 == (__int64 **)++v10 )
              goto LABEL_17;
          }
          return a1;
        }
      }
      return a1;
    }
    goto LABEL_17;
  }
  if ( sub_C8CA60(a1, (__int64)&qword_4F82400) )
    return a1;
LABEL_3:
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_19;
  v10 = *(void ***)(a1 + 8);
  v7 = *(unsigned int *)(a1 + 20);
  v6 = (__int64 **)&v10[v7];
  if ( v6 != (__int64 **)v10 )
    goto LABEL_7;
LABEL_17:
  if ( *(_DWORD *)(a1 + 16) > (unsigned int)v7 )
  {
    *(_DWORD *)(a1 + 20) = v7 + 1;
    *v6 = (__int64 *)&unk_4FDBCE8;
    ++*(_QWORD *)a1;
    return a1;
  }
LABEL_19:
  sub_C8CC70(a1, (__int64)&unk_4FDBCE8, (__int64)v6, v7, v8, v9);
  return a1;
}
