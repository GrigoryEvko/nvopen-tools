// Function: sub_29135A0
// Address: 0x29135a0
//
_QWORD *__fastcall sub_29135A0(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v3; // rsi
  _QWORD *v5; // r8
  __int64 v6; // rax
  unsigned __int64 v7; // r10
  __int64 v8; // rsi
  __int64 v9; // rcx
  unsigned __int64 *v10; // rdx

  v3 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
  if ( v3 > 0 )
  {
    v7 = *a3;
    while ( 1 )
    {
      v9 = v6 >> 1;
      v10 = &v5[(v6 >> 1) + (v6 & 0xFFFFFFFFFFFFFFFELL)];
      if ( v7 < *v10 )
        goto LABEL_5;
      if ( v7 > *v10 )
        goto LABEL_8;
      v8 = ((__int64)a3[2] >> 2) & 1;
      if ( (_BYTE)v8 == (((__int64)v10[2] >> 2) & 1) )
      {
        if ( a3[1] <= v10[1] )
        {
          v5 = v10 + 3;
          v6 = v6 - v9 - 1;
          goto LABEL_9;
        }
LABEL_5:
        v6 >>= 1;
        if ( v9 <= 0 )
          return v5;
      }
      else
      {
        if ( !(_BYTE)v8 )
          goto LABEL_5;
LABEL_8:
        v5 = v10 + 3;
        v6 = v6 - v9 - 1;
LABEL_9:
        if ( v6 <= 0 )
          return v5;
      }
    }
  }
  return v5;
}
