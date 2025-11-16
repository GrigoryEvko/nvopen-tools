// Function: sub_D542B0
// Address: 0xd542b0
//
__int64 __fastcall sub_D542B0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v6; // rsi
  __int64 v7; // rcx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  bool v10; // zf
  __int64 v11; // rdi
  __int64 v13; // rcx
  _QWORD *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // [rsp+0h] [rbp-40h] BYREF
  __int64 v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+10h] [rbp-30h]
  _QWORD *v28; // [rsp+18h] [rbp-28h]

  v6 = *a3;
  v7 = a3[1];
  v8 = (_QWORD *)a3[3];
  v9 = a3[2];
  v10 = a1[3] == a2[3];
  v25 = v6;
  v26 = v7;
  v11 = *a1;
  v27 = v9;
  v28 = v8;
  if ( !v10 )
  {
    if ( !(unsigned __int8)sub_D541A0(v11, a1[2], &v25) )
      return 0;
    v13 = a3[1];
    v14 = (_QWORD *)a3[3];
    v15 = ((a1[2] - *a1) >> 5) + ((*a3 - v13) >> 5);
    if ( v15 < 0 )
    {
      v16 = ~((unsigned __int64)~v15 >> 4);
    }
    else
    {
      if ( v15 <= 15 )
      {
        v18 = *a3 + a1[2] - *a1;
        v17 = a3[2];
        *a3 = v18;
LABEL_8:
        v19 = (__int64 *)(a1[3] + 8);
        if ( v19 == (__int64 *)a2[3] )
        {
LABEL_16:
          v27 = v17;
          v23 = a2[1];
          v28 = v14;
          v24 = *a2;
          v25 = v18;
          v26 = v13;
          return sub_D541A0(v23, v24, &v25);
        }
        while ( 1 )
        {
          while ( 1 )
          {
            v28 = v14;
            v25 = v18;
            v26 = v13;
            v27 = v17;
            if ( !(unsigned __int8)sub_D541A0(*v19, *v19 + 512, &v25) )
              return 0;
            v20 = *a3;
            v13 = a3[1];
            v14 = (_QWORD *)a3[3];
            v21 = ((*a3 - v13) >> 5) + 16;
            if ( v21 >= 0 )
              break;
            v22 = ~((unsigned __int64)~v21 >> 4);
LABEL_15:
            v14 += v22;
            ++v19;
            a3[3] = (__int64)v14;
            v13 = *v14;
            v17 = *v14 + 512LL;
            v18 = *v14 + 32 * (v21 - 16 * v22);
            a3[1] = *v14;
            a3[2] = v17;
            *a3 = v18;
            if ( (__int64 *)a2[3] == v19 )
              goto LABEL_16;
          }
          if ( v21 > 15 )
          {
            v22 = v21 >> 4;
            goto LABEL_15;
          }
          v18 = v20 + 512;
          v17 = a3[2];
          ++v19;
          *a3 = v20 + 512;
          if ( (__int64 *)a2[3] == v19 )
            goto LABEL_16;
        }
      }
      v16 = v15 >> 4;
    }
    v14 += v16;
    a3[3] = (__int64)v14;
    v13 = *v14;
    v17 = *v14 + 512LL;
    v18 = *v14 + 32 * (v15 - 16 * v16);
    a3[1] = *v14;
    a3[2] = v17;
    *a3 = v18;
    goto LABEL_8;
  }
  return sub_D541A0(v11, *a2, &v25);
}
