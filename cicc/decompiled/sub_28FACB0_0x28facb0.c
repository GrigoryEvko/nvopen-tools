// Function: sub_28FACB0
// Address: 0x28facb0
//
_QWORD *__fastcall sub_28FACB0(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v4; // r12
  _QWORD *v5; // r9
  _QWORD *v6; // rdi
  __int64 i; // r15
  unsigned __int64 v8; // rax
  signed __int64 v9; // r13
  _QWORD *v10; // r12
  unsigned __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v22; // [rsp+8h] [rbp-48h]
  signed __int64 v24; // [rsp+18h] [rbp-38h]

  v4 = a4;
  v5 = (_QWORD *)*a4;
  v6 = (_QWORD *)a4[1];
  v24 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a2) >> 3);
  if ( a3 - a2 > 0 )
  {
    for ( i = a3; ; i = v22 )
    {
      v8 = 0xAAAAAAAAAAAAAAABLL * (v5 - v6);
      if ( v5 == v6 )
      {
        v9 = 21;
        v10 = (_QWORD *)(*(_QWORD *)(a4[3] - 8LL) + 504LL);
      }
      else
      {
        v9 = 0xAAAAAAAAAAAAAAABLL * (v5 - v6);
        v10 = v5;
      }
      if ( v9 > v24 )
        v9 = v24;
      v22 = i - 24 * v9;
      v11 = 0xAAAAAAAAAAAAAAABLL * ((24 * v9) >> 3);
      if ( 24 * v9 > 0 )
      {
        do
        {
          v12 = *(_QWORD *)(i - 8);
          v13 = *(v10 - 1);
          i -= 24;
          v10 -= 3;
          if ( v12 != v13 )
          {
            if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
              sub_BD60C0(v10);
            v10[2] = v12;
            if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
              sub_BD73F0((__int64)v10);
          }
          --v11;
        }
        while ( v11 );
        v5 = (_QWORD *)*a4;
        v6 = (_QWORD *)a4[1];
        v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*a4 - (_QWORD)v6) >> 3);
      }
      v14 = v8 - v9;
      v15 = v14;
      if ( v14 < 0 )
      {
        v16 = ~(~v14 / 0x15uLL);
      }
      else
      {
        if ( v14 <= 20 )
        {
          v24 -= v9;
          v5 -= 3 * v9;
          *a4 = v5;
          if ( v24 <= 0 )
            goto LABEL_24;
          continue;
        }
        v16 = v14 / 21;
      }
      v24 -= v9;
      v17 = (_QWORD *)(a4[3] + 8 * v16);
      a4[3] = v17;
      v6 = (_QWORD *)*v17;
      v18 = *v17 + 504LL;
      a4[1] = v6;
      a4[2] = v18;
      v5 = &v6[3 * (v15 - 21 * v16)];
      *a4 = v5;
      if ( v24 <= 0 )
      {
LABEL_24:
        v4 = a4;
        break;
      }
    }
  }
  a1[2] = v4[2];
  v19 = v4[3];
  *a1 = v5;
  a1[3] = v19;
  a1[1] = v6;
  return a1;
}
