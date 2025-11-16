// Function: sub_1740980
// Address: 0x1740980
//
__int64 __fastcall sub_1740980(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v4; // r15
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int64 v28; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]

  result = *a1;
  v4 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = *(_BYTE *)(v4 + 23);
  if ( (*a1 & 4) != 0 )
  {
    if ( v5 < 0 )
    {
      v6 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
      v8 = v6 + v7;
      result = *(char *)(v4 + 23) >= 0 ? v8 >> 4 : (unsigned int)((v8 - sub_1648A40(v4)) >> 4);
      v9 = 0;
      v10 = 16LL * (unsigned int)result;
      if ( (_DWORD)result )
      {
        do
        {
          v11 = 0;
          if ( *(char *)(v4 + 23) < 0 )
            v11 = sub_1648A40(v4);
          v12 = (__int64 *)(v9 + v11);
          v9 += 16;
          v13 = *((unsigned int *)v12 + 2);
          v14 = *v12;
          v15 = *((unsigned int *)v12 + 3);
          v16 = 3 * v13;
          LODWORD(v13) = *(_DWORD *)(v4 + 20);
          v30 = v14;
          v16 *= 8;
          v28 = v4 + v16 - 24 * (v13 & 0xFFFFFFF);
          v29 = 0xAAAAAAAAAAAAAAABLL * ((24 * v15 - v16) >> 3);
          result = sub_1740580(a2, (__int64)&v28);
        }
        while ( v9 != v10 );
      }
    }
  }
  else if ( v5 < 0 )
  {
    result = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    v18 = result + v17;
    if ( *(char *)(v4 + 23) < 0 )
    {
      result = sub_1648A40(v4);
      v18 -= result;
    }
    v19 = v18 >> 4;
    if ( (_DWORD)v19 )
    {
      v20 = 0;
      v21 = 16LL * (unsigned int)v19;
      do
      {
        v22 = 0;
        if ( *(char *)(v4 + 23) < 0 )
          v22 = sub_1648A40(v4);
        v23 = (__int64 *)(v20 + v22);
        v20 += 16;
        v24 = *((unsigned int *)v23 + 2);
        v25 = *v23;
        v26 = *((unsigned int *)v23 + 3);
        v27 = 3 * v24;
        LODWORD(v24) = *(_DWORD *)(v4 + 20);
        v30 = v25;
        v27 *= 8;
        v28 = v4 + v27 - 24 * (v24 & 0xFFFFFFF);
        v29 = 0xAAAAAAAAAAAAAAABLL * ((24 * v26 - v27) >> 3);
        result = sub_1740580(a2, (__int64)&v28);
      }
      while ( v21 != v20 );
    }
  }
  return result;
}
