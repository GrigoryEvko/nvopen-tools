// Function: sub_2A62F90
// Address: 0x2a62f90
//
unsigned __int64 __fastcall sub_2A62F90(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  _QWORD *v8; // rax
  char *v9; // r14
  char *v10; // rsi
  __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // rdx
  _QWORD *v15; // rax
  unsigned __int64 *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // r15
  char *v19; // r14
  __int64 *v20; // r15
  size_t v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // r14
  __int64 v26; // rax
  const void *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // [rsp+8h] [rbp-38h]

  if ( *a2 == 6 )
  {
    result = *(unsigned int *)(a1 + 848);
    if ( !(_DWORD)result || *(_QWORD *)(*(_QWORD *)(a1 + 840) + 8 * result - 8) != a3 )
    {
      if ( result + 1 > *(unsigned int *)(a1 + 852) )
      {
        sub_C8D5F0(a1 + 840, (const void *)(a1 + 856), result + 1, 8u, a5, a6);
        result = *(unsigned int *)(a1 + 848);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 840) + 8 * result) = a3;
      ++*(_DWORD *)(a1 + 848);
    }
  }
  else if ( (_BYTE)qword_500BFA8 )
  {
    v8 = *(_QWORD **)(a1 + 1416);
    if ( v8 == (_QWORD *)(*(_QWORD *)(a1 + 1432) - 8LL) )
    {
      v9 = *(char **)(a1 + 1440);
      v10 = *(char **)(a1 + 1408);
      v11 = v9 - v10;
      v12 = (v9 - v10) >> 3;
      if ( (((__int64)v8 - *(_QWORD *)(a1 + 1424)) >> 3)
         + ((v12 - 1) << 6)
         + ((__int64)(*(_QWORD *)(a1 + 1400) - *(_QWORD *)(a1 + 1384)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
      v13 = *(_QWORD *)(a1 + 1368);
      v14 = *(_QWORD *)(a1 + 1376);
      if ( v14 - ((__int64)&v9[-v13] >> 3) <= 1 )
      {
        v18 = v12 + 2;
        if ( v14 <= 2 * (v12 + 2) )
        {
          v24 = 1;
          if ( v14 )
            v24 = *(_QWORD *)(a1 + 1376);
          v25 = v14 + v24 + 2;
          if ( v25 > 0xFFFFFFFFFFFFFFFLL )
            sub_4261EA(0xFFFFFFFFFFFFFFFLL, v10, v14);
          v26 = sub_22077B0(8 * v25);
          v27 = *(const void **)(a1 + 1408);
          v29 = v26;
          v20 = (__int64 *)(v26 + 8 * ((v25 - v18) >> 1));
          v28 = *(_QWORD *)(a1 + 1440) + 8LL;
          if ( (const void *)v28 != v27 )
            memmove(v20, v27, v28 - (_QWORD)v27);
          j_j___libc_free_0(*(_QWORD *)(a1 + 1368));
          *(_QWORD *)(a1 + 1376) = v25;
          *(_QWORD *)(a1 + 1368) = v29;
        }
        else
        {
          v19 = v9 + 8;
          v20 = (__int64 *)(v13 + 8 * ((v14 - v18) >> 1));
          v21 = v19 - v10;
          if ( v10 <= (char *)v20 )
          {
            if ( v19 != v10 )
              memmove((char *)v20 + v11 + 8 - v21, v10, v21);
          }
          else if ( v19 != v10 )
          {
            memmove(v20, v10, v21);
          }
        }
        *(_QWORD *)(a1 + 1408) = v20;
        v22 = *v20;
        v9 = (char *)v20 + v11;
        *(_QWORD *)(a1 + 1440) = (char *)v20 + v11;
        *(_QWORD *)(a1 + 1392) = v22;
        *(_QWORD *)(a1 + 1400) = v22 + 512;
        v23 = *(__int64 *)((char *)v20 + v11);
        *(_QWORD *)(a1 + 1424) = v23;
        *(_QWORD *)(a1 + 1432) = v23 + 512;
      }
      *((_QWORD *)v9 + 1) = sub_22077B0(0x200u);
      v15 = *(_QWORD **)(a1 + 1416);
      if ( v15 )
        *v15 = a3;
      v16 = (unsigned __int64 *)(*(_QWORD *)(a1 + 1440) + 8LL);
      *(_QWORD *)(a1 + 1440) = v16;
      result = *v16;
      v17 = *v16 + 512;
      *(_QWORD *)(a1 + 1424) = result;
      *(_QWORD *)(a1 + 1432) = v17;
      *(_QWORD *)(a1 + 1416) = result;
    }
    else
    {
      if ( v8 )
      {
        *v8 = a3;
        v8 = *(_QWORD **)(a1 + 1416);
      }
      result = (unsigned __int64)(v8 + 1);
      *(_QWORD *)(a1 + 1416) = result;
    }
  }
  else
  {
    result = *(unsigned int *)(a1 + 1456);
    if ( !(_DWORD)result || *(_QWORD *)(*(_QWORD *)(a1 + 1448) + 8 * result - 8) != a3 )
    {
      if ( result + 1 > *(unsigned int *)(a1 + 1460) )
      {
        sub_C8D5F0(a1 + 1448, (const void *)(a1 + 1464), result + 1, 8u, a5, a6);
        result = *(unsigned int *)(a1 + 1456);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 1448) + 8 * result) = a3;
      ++*(_DWORD *)(a1 + 1456);
    }
  }
  return result;
}
