// Function: sub_23B25D0
// Address: 0x23b25d0
//
__int64 __fastcall sub_23B25D0(__int64 a1, __int64 a2, __int64 a3, void *a4)
{
  void **v5; // rax
  void **v6; // rdx
  __int64 **v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 **v11; // rax
  __int64 v12; // rdx

  if ( *(_BYTE *)(a3 + 76) )
  {
    v5 = *(void ***)(a3 + 56);
    v6 = &v5[*(unsigned int *)(a3 + 68)];
    if ( v5 != v6 )
    {
      a4 = &unk_4FDE328;
      while ( *v5 != &unk_4FDE328 )
      {
        if ( v6 == ++v5 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4FDE328) )
  {
    return 1;
  }
LABEL_8:
  if ( *(_BYTE *)(a3 + 28) )
  {
    v8 = *(__int64 ***)(a3 + 8);
    v9 = (__int64)&v8[*(unsigned int *)(a3 + 20)];
    if ( v8 != (__int64 **)v9 )
    {
      while ( *v8 != &qword_4F82400 )
      {
        if ( (__int64 **)v9 == ++v8 )
          goto LABEL_15;
      }
      return 0;
    }
  }
  else if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
  {
    return 0;
  }
LABEL_15:
  if ( (unsigned __int8)sub_B19060(a3, (__int64)&unk_4FDE328, v9, (__int64)a4) )
    return 0;
  if ( !*(_BYTE *)(a3 + 28) )
  {
    if ( !sub_C8CA60(a3, (__int64)&qword_4F82400) )
      return (unsigned int)sub_B19060(a3, (__int64)&unk_4F82428, v12, v10) ^ 1;
    return 0;
  }
  v11 = *(__int64 ***)(a3 + 8);
  v12 = (__int64)&v11[*(unsigned int *)(a3 + 20)];
  if ( v11 == (__int64 **)v12 )
    return (unsigned int)sub_B19060(a3, (__int64)&unk_4F82428, v12, v10) ^ 1;
  while ( *v11 != &qword_4F82400 )
  {
    if ( (__int64 **)v12 == ++v11 )
      return (unsigned int)sub_B19060(a3, (__int64)&unk_4F82428, v12, v10) ^ 1;
  }
  return 0;
}
