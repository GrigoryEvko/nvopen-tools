// Function: sub_A04C00
// Address: 0xa04c00
//
__int64 *__fastcall sub_A04C00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        void (__fastcall *a7)(__int64, __int64),
        __int64 a8)
{
  __int64 v8; // rcx
  unsigned __int64 v9; // rbx
  int v10; // edx
  __int64 v11; // rbx
  const char *v13; // rax
  const char *v14; // rdx
  unsigned __int64 v15; // rdx
  const char *v16; // rax
  unsigned int v17; // [rsp+10h] [rbp-B0h]
  int v18; // [rsp+14h] [rbp-ACh]
  unsigned __int64 v19; // [rsp+18h] [rbp-A8h]
  __int64 v20; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v21; // [rsp+38h] [rbp-88h]
  unsigned __int64 v22; // [rsp+40h] [rbp-80h]
  __int64 v23; // [rsp+48h] [rbp-78h]
  int v24; // [rsp+50h] [rbp-70h]
  const char *v25; // [rsp+60h] [rbp-60h] BYREF
  char v26; // [rsp+68h] [rbp-58h]
  char v27; // [rsp+80h] [rbp-40h]
  char v28; // [rsp+81h] [rbp-3Fh]

  if ( a4 != 2 )
  {
    v28 = 1;
    v13 = "Invalid record: metadata strings layout";
LABEL_12:
    v25 = v13;
    v27 = 3;
    sub_A01DB0(a1, (__int64)&v25);
    return a1;
  }
  v8 = *(_QWORD *)a3;
  v18 = *(_QWORD *)a3;
  if ( !v18 )
  {
    v28 = 1;
    v13 = "Invalid record: metadata strings with no strings";
    goto LABEL_12;
  }
  v9 = *(unsigned int *)(a3 + 8);
  if ( a6 < v9 )
  {
    v28 = 1;
    v13 = "Invalid record: metadata strings corrupt offset";
    goto LABEL_12;
  }
  v10 = 0;
  v20 = a5;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v19 = a6 - v9;
  v21 = v9;
  v11 = a5 + v9;
  while ( 1 )
  {
    if ( !v10 && v21 <= v22 )
    {
      v28 = 1;
      v16 = "Invalid record: metadata strings bad length";
LABEL_19:
      v25 = v16;
      v27 = 3;
      sub_A01DB0(a1, (__int64)&v25);
      return a1;
    }
    sub_9CE2D0((__int64)&v25, (__int64)&v20, 6, v8);
    if ( (v26 & 1) != 0 )
    {
      v26 &= ~2u;
      v14 = v25;
      v25 = 0;
      v15 = (unsigned __int64)v14 & 0xFFFFFFFFFFFFFFFELL;
      if ( v15 )
      {
        *a1 = v15 | 1;
        return a1;
      }
    }
    else
    {
      v17 = (unsigned int)v25;
    }
    if ( v17 > v19 )
    {
      v28 = 1;
      v16 = "Invalid record: metadata strings truncated chars";
      goto LABEL_19;
    }
    a7(a8, v11);
    v19 -= v17;
    v11 += v17;
    if ( !--v18 )
      break;
    v10 = v24;
  }
  *a1 = 1;
  return a1;
}
