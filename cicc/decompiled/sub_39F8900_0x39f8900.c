// Function: sub_39F8900
// Address: 0x39f8900
//
__int64 __fastcall sub_39F8900(
        unsigned int (__fastcall *a1)(__m128i *, __int64),
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  unsigned int v7; // eax
  unsigned int v8; // r15d
  _QWORD *v9; // rdx
  __m128i v11; // [rsp+0h] [rbp-2A0h] BYREF
  _QWORD *v12; // [rsp+98h] [rbp-208h]
  char v13; // [rsp+C7h] [rbp-1D9h]
  _BYTE v14[24]; // [rsp+D8h] [rbp-1C8h]
  char v15[360]; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 v16; // [rsp+258h] [rbp-48h]
  __int64 retaddr; // [rsp+2A8h] [rbp+8h]

  sub_39F7A80(&v11, (__int64)&a7, retaddr);
  while ( 1 )
  {
    v7 = sub_39F7420(&v11, v15);
    v8 = v7;
    if ( v7 )
    {
      if ( v7 != 5 )
        return 3;
    }
    if ( a1(&v11, a2) )
      return 3;
    if ( v8 == 5 )
      return v8;
    sub_39F6770(&v11, (__int64)v15);
    if ( *(_DWORD *)&v15[16 * v16 + 8] == 6 )
    {
      v12 = 0;
    }
    else
    {
      if ( (int)v16 > 17 )
        goto LABEL_16;
      v9 = (_QWORD *)v11.m128i_i64[(int)v16];
      if ( (v13 & 0x40) == 0 || !v14[(int)v16] )
      {
        if ( byte_5057700[(int)v16] != 8 )
LABEL_16:
          abort();
        v9 = (_QWORD *)*v9;
      }
      v12 = v9;
    }
  }
}
