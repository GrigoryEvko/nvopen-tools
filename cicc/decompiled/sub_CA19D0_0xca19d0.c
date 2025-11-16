// Function: sub_CA19D0
// Address: 0xca19d0
//
__int64 __fastcall sub_CA19D0(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // r13d
  unsigned __int64 v4; // rbx
  _BYTE *v5; // r15
  unsigned int v6; // eax
  unsigned int v7; // r15d
  char *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  char *v11; // rcx
  char *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  char *v15; // rcx
  unsigned int v17; // [rsp-4Ch] [rbp-4Ch] BYREF
  _BYTE *v18; // [rsp-48h] [rbp-48h] BYREF
  unsigned int *v19; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 )
  {
    v3 = 0;
    v4 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v5 = (_BYTE *)(a1 + v4);
          v6 = sub_F03780(*(unsigned __int8 *)(a1 + v4));
          if ( v6 != 1 )
            break;
          if ( (unsigned __int8)(*v5 - 32) > 0x5Eu )
            return 0xFFFFFFFFLL;
          ++v4;
          ++v3;
          if ( a2 <= v4 )
            return v3;
        }
        if ( !v6 )
          return 4294967294LL;
        v4 += v6;
        if ( v4 > a2 )
          return 4294967294LL;
        v18 = v5;
        v19 = &v17;
        if ( (unsigned int)sub_F03810(&v18, &v5[v6], &v19, &v18, 0) )
          return 4294967294LL;
        v7 = v17;
        if ( !sub_CA1970(v17) )
          return 0xFFFFFFFFLL;
        v8 = (char *)&unk_3F67F80;
        v9 = 343;
        do
        {
          while ( 1 )
          {
            v10 = v9 >> 1;
            v11 = &v8[8 * (v9 >> 1)];
            if ( v7 <= *((_DWORD *)v11 + 1) )
              break;
            v8 = v11 + 8;
            v9 = v9 - v10 - 1;
            if ( v9 <= 0 )
              goto LABEL_12;
          }
          v9 >>= 1;
        }
        while ( v10 > 0 );
LABEL_12:
        if ( v8 == (char *)&unk_3F68A38 || v7 < *(_DWORD *)v8 )
          break;
LABEL_21:
        if ( a2 <= v4 )
          return v3;
      }
      v12 = (char *)&unk_3F67C60;
      v13 = 100;
      do
      {
        while ( 1 )
        {
          v14 = v13 >> 1;
          v15 = &v12[8 * (v13 >> 1)];
          if ( v7 <= *((_DWORD *)v15 + 1) )
            break;
          v12 = v15 + 8;
          v13 = v13 - v14 - 1;
          if ( v13 <= 0 )
            goto LABEL_18;
        }
        v13 >>= 1;
      }
      while ( v14 > 0 );
LABEL_18:
      if ( v12 != (char *)&unk_3F67F80 && v7 >= *(_DWORD *)v12 )
      {
        v3 += 2;
        goto LABEL_21;
      }
      ++v3;
      if ( a2 <= v4 )
        return v3;
    }
  }
  return 0;
}
