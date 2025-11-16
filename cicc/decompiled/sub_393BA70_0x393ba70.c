// Function: sub_393BA70
// Address: 0x393ba70
//
unsigned __int64 *__fastcall sub_393BA70(
        unsigned __int64 *a1,
        __int64 a2,
        size_t **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r12
  size_t *v7; // r13
  size_t *v8; // rbx
  size_t *v9; // r15
  size_t *v10; // rdx
  unsigned __int64 v11; // rax
  size_t *i; // [rsp+18h] [rbp-48h]
  __int64 v14[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = a2;
  v7 = a3[2];
  v8 = *a3;
  v9 = a3[1];
  for ( i = a3[6]; i != v7; v8 = (size_t *)a3 )
  {
    if ( v9 )
    {
      a2 = v6;
      sub_1696B90(v14, v6, v8 + 3, v8[1]);
      v10 = v8 + 3;
      v11 = v14[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v14[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_7;
    }
    else
    {
      a2 = v6;
      sub_1696B90(v14, v6, (char *)v8 + 26, *(size_t *)((char *)v8 + 10));
      v11 = v14[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v14[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
LABEL_7:
        *a1 = v11 | 1;
        return a1;
      }
      v9 = (size_t *)*(unsigned __int16 *)v8;
      v8 = (size_t *)((char *)v8 + 2);
      v10 = v8 + 3;
    }
    a3 = (size_t **)((char *)v10 + v8[2] + v8[1]);
    v9 = (size_t *)((char *)v9 - 1);
    v7 = (size_t *)((char *)v7 - 1);
  }
  sub_16977B0(v6, a2, (__int64)a3, a4, a5, a6);
  *a1 = 1;
  return a1;
}
