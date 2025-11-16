// Function: sub_232ED20
// Address: 0x232ed20
//
__int64 __fastcall sub_232ED20(__int64 a1, __int64 a2, __int64 a3, void *a4)
{
  void **v5; // rax
  __int64 v6; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx

  if ( *(_BYTE *)(a3 + 76) )
  {
    v5 = *(void ***)(a3 + 56);
    v6 = (__int64)&v5[*(unsigned int *)(a3 + 68)];
    if ( v5 != (void **)v6 )
    {
      a4 = &unk_4F8ED68;
      while ( *v5 != &unk_4F8ED68 )
      {
        if ( (void **)v6 == ++v5 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4F8ED68) )
  {
    return 1;
  }
LABEL_8:
  if ( (unsigned __int8)sub_B19060(a3, (__int64)&qword_4F82400, v6, (__int64)a4)
    || (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F8ED68, v8, v9)
    || (unsigned __int8)sub_B19060(a3, (__int64)&qword_4F82400, v10, v11) )
  {
    return 0;
  }
  else
  {
    return (unsigned int)sub_B19060(a3, (__int64)&unk_4F82420, v12, v13) ^ 1;
  }
}
