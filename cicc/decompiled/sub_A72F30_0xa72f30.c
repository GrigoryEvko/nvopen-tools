// Function: sub_A72F30
// Address: 0xa72f30
//
__int64 __fastcall sub_A72F30(__int64 a1, __int64 a2, char a3)
{
  char v4; // al
  int v5; // r14d
  __int64 result; // rax
  unsigned __int64 v7; // rbx
  __int64 v8; // rbx
  int v9; // ebx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  const void *v13; // r14
  const void *v14; // rax
  size_t v15; // rdx
  const void *v16; // r13
  size_t v17; // rdx
  size_t v18; // rbx
  const void *v19; // rax
  size_t v20; // rdx
  size_t v21; // r12
  int v22; // eax
  size_t v23; // rdx

  if ( a1 == a2 )
    return 0;
  v4 = *(_BYTE *)(a2 + 8);
  if ( *(_BYTE *)(a1 + 8) == 2 )
  {
    if ( v4 == 2 )
    {
      if ( a3
        || (v10 = sub_A71FC0(a2), v12 = v11, v13 = (const void *)v10, v14 = (const void *)sub_A71FC0(a1), v15 != v12)
        || v15 && memcmp(v14, v13, v15) )
      {
        v16 = (const void *)sub_A71FC0(a1);
        v18 = v17;
        v19 = (const void *)sub_A71FC0(a2);
      }
      else
      {
        v16 = (const void *)sub_A72230(a1);
        v18 = v23;
        v19 = (const void *)sub_A72230(a2);
      }
      v21 = v20;
      if ( v18 <= v20 )
        v20 = v18;
      if ( v20 )
      {
        v22 = memcmp(v16, v19, v20);
        if ( v22 )
          return (v22 >> 31) | 1u;
      }
      result = 0;
      if ( v18 != v21 )
        return v18 < v21 ? -1 : 1;
      return result;
    }
    return 1;
  }
  if ( v4 == 2 )
    return 0xFFFFFFFFLL;
  v5 = sub_A71AD0(a1);
  if ( v5 != (unsigned int)sub_A71AD0(a2) )
  {
    v9 = sub_A71AD0(a1);
    if ( v9 >= (int)sub_A71AD0(a2) )
      return 1;
    return 0xFFFFFFFFLL;
  }
  result = 0;
  if ( !a3 )
  {
    v7 = sub_A71B70(a1);
    if ( v7 >= sub_A71B70(a2) )
    {
      v8 = sub_A71B70(a1);
      return v8 != sub_A71B70(a2);
    }
    return 0xFFFFFFFFLL;
  }
  return result;
}
