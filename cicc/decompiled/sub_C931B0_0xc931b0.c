// Function: sub_C931B0
// Address: 0xc931b0
//
__int64 __fastcall sub_C931B0(__int64 *a1, _WORD *a2, size_t a3, unsigned __int64 a4)
{
  unsigned __int64 v5; // rdx
  __int64 result; // rax
  size_t v7; // rdx
  __int64 v8; // r15
  _WORD *v9; // r12
  unsigned __int64 v10; // r14
  __int64 v11; // rcx
  size_t v12; // rdx
  unsigned int v13; // eax
  char v14; // cl
  __int64 v15; // rbx
  int v16; // eax
  void *v17; // rax
  char v18; // [rsp+7h] [rbp-139h]
  size_t v19; // [rsp+8h] [rbp-138h]
  _OWORD v20[19]; // [rsp+10h] [rbp-130h]

  v5 = a1[1];
  if ( a4 <= v5 )
  {
    result = a4;
    if ( !a3 )
      return result;
    v7 = v5 - a4;
    if ( v7 >= a3 )
    {
      v8 = *a1;
      v9 = (_WORD *)(*a1 + a4);
      if ( a3 != 1 )
      {
        v10 = (unsigned __int64)v9 + v7 - a3 + 1;
        if ( a3 == 2 )
        {
          while ( *a2 != *v9 )
          {
            v9 = (_WORD *)((char *)v9 + 1);
            if ( v10 <= (unsigned __int64)v9 )
              return -1;
          }
        }
        else if ( v7 <= 0xF || a3 > 0xFF )
        {
          while ( memcmp(v9, a2, a3) )
          {
            v9 = (_WORD *)((char *)v9 + 1);
            if ( v10 <= (unsigned __int64)v9 )
              return -1;
          }
        }
        else
        {
          v20[0] = __PAIR128__(0x101010101010101LL, 0x101010101010101LL) * (unsigned __int8)a3;
          v20[1] = v20[0];
          v20[2] = v20[0];
          v20[3] = v20[0];
          v20[4] = v20[0];
          v20[5] = v20[0];
          v20[6] = v20[0];
          v20[7] = v20[0];
          v20[8] = v20[0];
          v20[9] = v20[0];
          v20[10] = v20[0];
          v20[11] = v20[0];
          v20[12] = v20[0];
          v20[13] = v20[0];
          v20[14] = v20[0];
          v11 = 0;
          v12 = a3 - 1;
          v20[15] = v20[0];
          v13 = 0;
          do
          {
            *((_BYTE *)v20 + *((unsigned __int8 *)a2 + v11)) = a3 - 1 - v13++;
            v11 = v13;
          }
          while ( v13 != v12 );
          v14 = *((_BYTE *)a2 + a3 - 1);
          while ( 1 )
          {
            v15 = *((unsigned __int8 *)v9 + v12);
            if ( (_BYTE)v15 == v14 )
            {
              v18 = v14;
              v19 = v12;
              v16 = memcmp(v9, a2, v12);
              v12 = v19;
              v14 = v18;
              if ( !v16 )
                break;
            }
            v9 = (_WORD *)((char *)v9 + *((unsigned __int8 *)v20 + v15));
            if ( v10 <= (unsigned __int64)v9 )
              return -1;
          }
        }
        v17 = v9;
        return (__int64)v17 - v8;
      }
      v17 = memchr((const void *)(*a1 + a4), *(char *)a2, v7);
      if ( v17 )
        return (__int64)v17 - v8;
    }
  }
  return -1;
}
