// Function: sub_28FAF30
// Address: 0x28faf30
//
_QWORD *__fastcall sub_28FAF30(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // r15
  unsigned __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // r14
  signed __int64 v10; // r13
  __int64 v11; // r12
  unsigned __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // rax
  _QWORD **v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 v23; // [rsp+28h] [rbp-38h]

  v4 = a3 - a2;
  v5 = a4;
  v6 = (_QWORD *)*a4;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a2) >> 3);
  v22 = a2;
  v8 = a4[2];
  v23 = v7;
  if ( v4 > 0 )
  {
    while ( 1 )
    {
      v9 = v22;
      v10 = 0xAAAAAAAAAAAAAAABLL * ((v8 - (__int64)v6) >> 3);
      if ( v10 > v23 )
        v10 = v23;
      v22 += 24 * v10;
      if ( 24 * v10 > 0 )
      {
        v21 = (_QWORD **)v5;
        v11 = v9;
        v12 = 0xAAAAAAAAAAAAAAABLL * ((24 * v10) >> 3);
        do
        {
          v13 = *(_QWORD *)(v11 + 16);
          v14 = v6[2];
          if ( v13 != v14 )
          {
            if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
              sub_BD60C0(v6);
            v6[2] = v13;
            if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
              sub_BD73F0((__int64)v6);
          }
          v11 += 24;
          v6 += 3;
          --v12;
        }
        while ( v12 );
        v5 = v21;
        v6 = *v21;
      }
      v15 = v10 - 0x5555555555555555LL * (((__int64)v6 - v5[1]) >> 3);
      if ( v15 < 0 )
        break;
      if ( v15 > 20 )
      {
        v18 = v15 / 21;
LABEL_21:
        v19 = (__int64 *)(v5[3] + 8 * v18);
        v5[3] = v19;
        v20 = *v19;
        v8 = v20 + 504;
        v6 = (_QWORD *)(v20 + 24 * (v15 - 21 * v18));
        v5[1] = v20;
        v5[2] = v20 + 504;
        *v5 = v6;
        goto LABEL_18;
      }
      v6 += 3 * v10;
      v8 = v5[2];
      *v5 = v6;
LABEL_18:
      v23 -= v10;
      if ( v23 <= 0 )
        goto LABEL_19;
    }
    v18 = ~(~v15 / 0x15uLL);
    goto LABEL_21;
  }
LABEL_19:
  a1[1] = v5[1];
  v16 = v5[3];
  *a1 = v6;
  a1[3] = v16;
  a1[2] = v8;
  return a1;
}
