// Function: sub_155E7C0
// Address: 0x155e7c0
//
char __fastcall sub_155E7C0(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned __int8 v3; // dl
  int v4; // eax
  __int64 v5; // rax
  size_t v6; // rdx
  size_t v7; // r14
  const void *v8; // r15
  const void *v9; // rax
  __int64 v10; // rdx
  const void *v11; // r14
  size_t v12; // rdx
  size_t v13; // r13
  size_t v14; // rdx
  const void *v15; // rdi
  size_t v16; // r12
  unsigned int v17; // eax
  int v18; // r12d
  const void *v19; // r14
  size_t v20; // rdx
  size_t v21; // r13
  size_t v22; // rdx
  const void *v23; // rdi
  size_t v24; // r12
  int v25; // r14d
  unsigned __int64 v26; // r12

  v2 = *(_BYTE *)(a1 + 16);
  v3 = *(_BYTE *)(a2 + 16);
  if ( !v2 )
  {
    if ( v3 )
    {
      LOBYTE(v4) = 1;
      if ( (unsigned __int8)(v3 - 1) <= 1u )
        return v4;
      goto LABEL_4;
    }
LABEL_16:
    v18 = sub_155D400(a1);
    LOBYTE(v4) = v18 < (int)sub_155D400(a2);
    return v4;
  }
  if ( v2 != 1 )
  {
LABEL_4:
    LOBYTE(v4) = 0;
    if ( v3 <= 1u )
      return v4;
    goto LABEL_10;
  }
  LOBYTE(v4) = 0;
  if ( !v3 )
    return v4;
  if ( v3 == 1 )
  {
    v25 = sub_155D400(a1);
    if ( v25 == (unsigned int)sub_155D400(a2) )
    {
      v26 = sub_155D4A0(a1);
      LOBYTE(v4) = v26 < sub_155D4A0(a2);
      return v4;
    }
    goto LABEL_16;
  }
  LOBYTE(v4) = 1;
  if ( v3 == 2 )
    return v4;
LABEL_10:
  v5 = sub_155D7C0(a2);
  v7 = v6;
  v8 = (const void *)v5;
  v9 = (const void *)sub_155D7C0(a1);
  if ( v7 == v10 && (!v7 || !memcmp(v9, v8, v7)) )
  {
    v19 = (const void *)sub_155D8A0(a2);
    v21 = v20;
    v23 = (const void *)sub_155D8A0(a1);
    v24 = v22;
    if ( v22 > v21 )
    {
      LOBYTE(v4) = 0;
      if ( !v21 )
        return v4;
      v17 = memcmp(v23, v19, v21);
      if ( !v17 )
      {
LABEL_23:
        LOBYTE(v4) = v24 < v21;
        return v4;
      }
    }
    else if ( !v22 || (v17 = memcmp(v23, v19, v22)) == 0 )
    {
      LOBYTE(v4) = 0;
      if ( v24 == v21 )
        return v4;
      goto LABEL_23;
    }
    return v17 >> 31;
  }
  v11 = (const void *)sub_155D7C0(a2);
  v13 = v12;
  v15 = (const void *)sub_155D7C0(a1);
  v16 = v14;
  if ( v13 < v14 )
  {
    LOBYTE(v4) = 0;
    if ( !v13 )
      return v4;
    v17 = memcmp(v15, v11, v13);
    if ( v17 )
      return v17 >> 31;
  }
  else
  {
    if ( v14 )
    {
      v17 = memcmp(v15, v11, v14);
      if ( v17 )
        return v17 >> 31;
    }
    LOBYTE(v4) = 0;
    if ( v13 == v16 )
      return v4;
  }
  LOBYTE(v4) = v13 > v16;
  return v4;
}
