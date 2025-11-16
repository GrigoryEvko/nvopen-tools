// Function: sub_3702660
// Address: 0x3702660
//
unsigned __int64 *__fastcall sub_3702660(
        unsigned __int64 *a1,
        _QWORD *a2,
        unsigned __int64 *a3,
        const __m128i *a4,
        __int64 a5,
        __int64 a6)
{
  bool v9; // zf
  __int64 v10; // rsi
  unsigned __int64 v11; // rdi
  bool v13; // cc
  __int64 v14; // [rsp+8h] [rbp-28h] BYREF
  unsigned __int64 v15; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-18h]
  char v17; // [rsp+1Ch] [rbp-14h]

  v9 = a2[7] == 0;
  v10 = a2[5];
  if ( v9 )
  {
    if ( a2[6] && !v10 )
    {
      sub_3702300(&v15, (__int64)a2, a3);
      if ( (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = v15 & 0xFFFFFFFFFFFFFFFELL | 1;
        return a1;
      }
      goto LABEL_6;
    }
  }
  else if ( !v10 && !a2[6] )
  {
    sub_3701BE0(a2, a3, a4, (__int64)a4, a5, a6);
LABEL_6:
    *a1 = 1;
    return a1;
  }
  v17 = 0;
  v16 = 1;
  v15 = 0;
  sub_3708A50(&v14, v10, &v15);
  if ( (v14 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v11 = v15;
    if ( v16 <= 0x40 )
    {
      *a3 = v15;
    }
    else
    {
      *a3 = *(_QWORD *)v15;
      j_j___libc_free_0_0(v11);
    }
    goto LABEL_6;
  }
  v13 = v16 <= 0x40;
  *a1 = v14 & 0xFFFFFFFFFFFFFFFELL | 1;
  if ( !v13 && v15 )
    j_j___libc_free_0_0(v15);
  return a1;
}
