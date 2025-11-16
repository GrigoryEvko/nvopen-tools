// Function: sub_2C6F330
// Address: 0x2c6f330
//
char __fastcall sub_2C6F330(__int64 a1, __int64 a2, __int64 a3)
{
  void **v4; // rax
  void **v5; // rdx
  char result; // al
  __int64 **v7; // rdx
  __int64 **v8; // rcx
  __int64 **v9; // rax
  void **v10; // rcx
  void **v11; // rcx
  void **v12; // rdx
  __int64 **v13; // rax

  if ( *(_BYTE *)(a3 + 76) )
  {
    v4 = *(void ***)(a3 + 56);
    v5 = &v4[*(unsigned int *)(a3 + 68)];
    if ( v4 != v5 )
    {
      while ( *v4 != &unk_5010CC8 )
      {
        if ( v5 == ++v4 )
          goto LABEL_9;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_5010CC8) )
  {
    return 1;
  }
LABEL_9:
  if ( *(_BYTE *)(a3 + 28) )
  {
    v7 = *(__int64 ***)(a3 + 8);
    v8 = &v7[*(unsigned int *)(a3 + 20)];
    if ( v7 != v8 )
    {
      v9 = *(__int64 ***)(a3 + 8);
      while ( *v9 != &qword_4F82400 )
      {
        if ( v8 == ++v9 )
          goto LABEL_25;
      }
      return 0;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
    return 0;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v7 = *(__int64 ***)(a3 + 8);
    v9 = &v7[*(unsigned int *)(a3 + 20)];
    if ( v9 == v7 )
      return 1;
LABEL_25:
    v11 = (void **)v7;
    while ( *v11 != &unk_5010CC8 )
    {
      if ( ++v11 == (void **)v9 )
        goto LABEL_20;
    }
    return 0;
  }
  if ( sub_C8CA60(a3, (__int64)&unk_5010CC8) )
    return 0;
  if ( !*(_BYTE *)(a3 + 28) )
  {
    if ( !sub_C8CA60(a3, (__int64)&qword_4F82400) )
    {
      if ( *(_BYTE *)(a3 + 28) )
      {
        v10 = *(void ***)(a3 + 8);
        v9 = (__int64 **)&v10[*(unsigned int *)(a3 + 20)];
        if ( v9 == (__int64 **)v10 )
          return 1;
LABEL_40:
        v12 = v10;
        while ( *v10 != &unk_4F82420 )
        {
          if ( ++v10 == (void **)v9 )
            goto LABEL_34;
        }
        return 0;
      }
      if ( !sub_C8CA60(a3, (__int64)&unk_4F82420) )
      {
        if ( *(_BYTE *)(a3 + 28) )
        {
          v12 = *(void ***)(a3 + 8);
          v10 = &v12[*(unsigned int *)(a3 + 20)];
LABEL_34:
          if ( v12 == v10 )
            return 1;
          v13 = (__int64 **)v12;
          while ( *v13 != &qword_4F82400 )
          {
            if ( ++v13 == (__int64 **)v10 )
              goto LABEL_50;
          }
          return 0;
        }
        if ( !sub_C8CA60(a3, (__int64)&qword_4F82400) )
        {
          result = *(_BYTE *)(a3 + 28);
          if ( !result )
            return sub_C8CA60(a3, (__int64)&unk_4F82408) == 0;
          v12 = *(void ***)(a3 + 8);
          v10 = &v12[*(unsigned int *)(a3 + 20)];
          if ( v12 != v10 )
          {
LABEL_50:
            while ( *v12 != &unk_4F82408 )
            {
              if ( v10 == ++v12 )
                return 1;
            }
            return 0;
          }
          return result;
        }
      }
    }
    return 0;
  }
  v7 = *(__int64 ***)(a3 + 8);
  v9 = &v7[*(unsigned int *)(a3 + 20)];
  if ( v9 == v7 )
    return 1;
LABEL_20:
  v10 = (void **)v7;
  while ( *v7 != &qword_4F82400 )
  {
    if ( v9 == ++v7 )
      goto LABEL_40;
  }
  return 0;
}
