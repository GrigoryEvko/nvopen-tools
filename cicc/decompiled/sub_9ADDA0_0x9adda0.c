// Function: sub_9ADDA0
// Address: 0x9adda0
//
char __fastcall sub_9ADDA0(
        unsigned __int8 a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        __int64 a6,
        unsigned __int64 *a7,
        unsigned __int64 *a8,
        int a9,
        __m128i *a10)
{
  bool v12; // al
  char result; // al
  bool v14; // cc
  unsigned int v15; // eax
  unsigned __int64 v16; // rdi
  int v17; // [rsp+4h] [rbp-5Ch]
  int v18; // [rsp+4h] [rbp-5Ch]
  unsigned __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-48h]
  unsigned __int64 v22; // [rsp+20h] [rbp-40h]
  int v23; // [rsp+28h] [rbp-38h]

  sub_9AB8E0(a3, a6, a7, a9 + 1, a10);
  if ( *((_DWORD *)a7 + 2) <= 0x40u )
  {
    v12 = *a7 == 0;
  }
  else
  {
    v17 = *((_DWORD *)a7 + 2);
    v12 = v17 == (unsigned int)sub_C444A0(a7);
  }
  if ( !v12
    || (*((_DWORD *)a7 + 6) <= 0x40u
      ? (result = a7[2] == 0)
      : (v18 = *((_DWORD *)a7 + 6), result = v18 == (unsigned int)sub_C444A0(a7 + 2)),
        !result || a4 || a5) )
  {
    sub_9AB8E0(a2, a6, a8, a9 + 1, a10);
    sub_C70430(&v20, a1, a4, a5, a8, a7);
    if ( *((_DWORD *)a7 + 2) > 0x40u && *a7 )
      j_j___libc_free_0_0(*a7);
    v14 = *((_DWORD *)a7 + 6) <= 0x40u;
    *a7 = v20;
    v15 = v21;
    v21 = 0;
    *((_DWORD *)a7 + 2) = v15;
    if ( v14 || (v16 = a7[2]) == 0 )
    {
      a7[2] = v22;
      result = v23;
      *((_DWORD *)a7 + 6) = v23;
    }
    else
    {
      j_j___libc_free_0_0(v16);
      v14 = v21 <= 0x40;
      a7[2] = v22;
      result = v23;
      *((_DWORD *)a7 + 6) = v23;
      if ( !v14 )
      {
        if ( v20 )
          return j_j___libc_free_0_0(v20);
      }
    }
  }
  return result;
}
