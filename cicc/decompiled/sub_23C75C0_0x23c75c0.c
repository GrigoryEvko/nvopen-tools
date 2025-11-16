// Function: sub_23C75C0
// Address: 0x23c75c0
//
__int64 __fastcall sub_23C75C0(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        const void *a4,
        size_t a5,
        __int64 a6,
        char *a7,
        unsigned __int64 a8,
        const void *a9,
        size_t a10)
{
  int v10; // ecx
  __int64 *v12; // rsi
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // r15
  __int64 *v18; // r13
  __int64 v19; // r15
  __int64 result; // rax
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 *v27; // [rsp+18h] [rbp-38h]

  v10 = *(_DWORD *)(a1 + 8);
  if ( !v10 )
    return 0;
  v12 = *(__int64 **)a1;
  v14 = **(_QWORD **)a1;
  if ( v14 != -8 && v14 )
  {
    v17 = *(__int64 **)a1;
  }
  else
  {
    v15 = v12 + 1;
    do
    {
      do
      {
        v16 = *v15;
        v17 = v15++;
      }
      while ( v16 == -8 );
    }
    while ( !v16 );
  }
  v27 = &v12[v10];
  if ( v27 == v17 )
    return 0;
  v18 = v17;
  while ( 1 )
  {
    v19 = *v18;
    if ( (unsigned int)sub_23C6E80(*(__int64 ***)(*v18 + 8), a2, a3) )
    {
      result = sub_23C74C0(a1, v19 + 16, a4, a5, a7, a8, a9, a10);
      if ( (_DWORD)result )
        break;
    }
    v21 = v18[1];
    v22 = v18 + 1;
    if ( v21 != -8 && v21 )
    {
      ++v18;
      if ( v22 == v27 )
        return 0;
    }
    else
    {
      v23 = v18 + 2;
      do
      {
        do
        {
          v24 = *v23;
          v18 = v23++;
        }
        while ( v24 == -8 );
      }
      while ( !v24 );
      if ( v18 == v27 )
        return 0;
    }
  }
  return result;
}
