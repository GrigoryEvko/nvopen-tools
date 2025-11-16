// Function: sub_2509010
// Address: 0x2509010
//
__int64 *__fastcall sub_2509010(__int64 *a1, int a2)
{
  int v2; // eax
  _BOOL8 v3; // r8
  __int64 *v5; // rdi
  int v6; // r13d
  unsigned int v7; // esi
  unsigned int v8; // r13d
  unsigned int v9; // r12d
  unsigned __int64 v10; // rdx
  int v11; // r15d
  unsigned __int64 v12; // rcx
  int v13; // ebx
  unsigned int v14; // ebx
  _BYTE *v15; // rcx
  unsigned int v16; // eax
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v22; // r13
  unsigned __int64 v23; // rsi
  _BOOL8 v24; // [rsp+8h] [rbp-38h]
  _BOOL8 v25; // [rsp+8h] [rbp-38h]
  _BOOL8 v26; // [rsp+8h] [rbp-38h]
  _BOOL8 v27; // [rsp+8h] [rbp-38h]

  v2 = a2 >> 31;
  v3 = a2 < 0;
  v5 = a1 + 2;
  v6 = a2 ^ (a2 >> 31);
  v7 = (unsigned int)a2 >> 31;
  v8 = v6 - v2;
  if ( v8 > 9 )
  {
    if ( v8 <= 0x63 )
    {
      *a1 = (__int64)v5;
      v25 = v3;
      sub_2240A50(a1, v7 + 2, 45);
      v15 = (_BYTE *)(v25 + *a1);
    }
    else
    {
      if ( v8 <= 0x3E7 )
      {
        v14 = 2;
        v11 = 3;
        v9 = v8;
      }
      else
      {
        v9 = v8;
        v10 = v8;
        if ( v8 <= 0x270F )
        {
          v14 = 3;
          v11 = 4;
        }
        else
        {
          v11 = 1;
          do
          {
            v12 = v10;
            v13 = v11;
            v11 += 4;
            v10 /= 0x2710u;
            if ( v12 <= 0x1869F )
            {
              v14 = v13 + 3;
              goto LABEL_11;
            }
            if ( (unsigned int)v10 <= 0x63 )
            {
              *a1 = (__int64)v5;
              v23 = v13 + v7 + 5;
              v26 = v3;
              v14 = v11;
              sub_2240A50(a1, v23, 45);
              v15 = (_BYTE *)(v26 + *a1);
              goto LABEL_14;
            }
            if ( (unsigned int)v10 <= 0x3E7 )
            {
              v11 = v13 + 6;
              v14 = v13 + 5;
              goto LABEL_11;
            }
          }
          while ( (unsigned int)v10 > 0x270F );
          v11 = v13 + 7;
          v14 = v13 + 6;
        }
      }
LABEL_11:
      *a1 = (__int64)v5;
      v24 = v3;
      sub_2240A50(a1, v11 + v7, 45);
      v15 = (_BYTE *)(v24 + *a1);
LABEL_14:
      while ( 1 )
      {
        v16 = v8 - 100 * (v9 / 0x64);
        v17 = v8;
        v8 = v9 / 0x64;
        v18 = 2 * v16;
        v19 = (unsigned int)(v18 + 1);
        LOBYTE(v18) = a00010203040506[v18];
        v15[v14] = a00010203040506[v19];
        v20 = v14 - 1;
        v14 -= 2;
        v15[v20] = v18;
        if ( v17 <= 0x270F )
          break;
        v9 /= 0x64u;
      }
      if ( v17 <= 0x3E7 )
        goto LABEL_16;
    }
    v22 = 2 * v8;
    v15[1] = a00010203040506[(unsigned int)(v22 + 1)];
    *v15 = a00010203040506[v22];
    return a1;
  }
  *a1 = (__int64)v5;
  v27 = v3;
  sub_2240A50(a1, v7 + 1, 45);
  v15 = (_BYTE *)(v27 + *a1);
LABEL_16:
  *v15 = v8 + 48;
  return a1;
}
