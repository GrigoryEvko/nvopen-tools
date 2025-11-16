// Function: sub_D4B950
// Address: 0xd4b950
//
unsigned __int8 **__fastcall sub_D4B950(
        unsigned __int8 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 *v12; // rbx
  __int64 *v13; // rdx
  __int64 *v14; // r14
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v16; // rsi
  unsigned __int8 *v17; // rsi
  __int64 v19; // rdi
  unsigned __int8 *v20; // rsi
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // rsi
  __int64 v23; // rdi
  unsigned __int8 *v24; // rsi
  unsigned __int8 *v25; // rsi
  unsigned __int8 *v26; // rsi
  unsigned __int8 *v27; // rsi
  unsigned __int8 *v28; // rsi
  unsigned __int8 *v29; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int8 *v30; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int8 *v31[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = sub_D49300(a2, a2, a3, a4, a5, a6);
  if ( !v8
    || ((v29 = 0, v9 = *(_BYTE *)(v8 - 16), (v9 & 2) != 0)
      ? (v11 = *(_QWORD *)(v8 - 32), v10 = *(unsigned int *)(v8 - 24))
      : (v10 = (*(_WORD *)(v8 - 16) >> 6) & 0xF, v11 = v8 - 8LL * ((v9 >> 2) & 0xF) - 16),
        v12 = (__int64 *)sub_D46550(v11, v10, 1),
        v14 = v13,
        v12 == v13) )
  {
LABEL_22:
    v19 = sub_D4B130(a2);
    if ( v19 )
    {
      v20 = *(unsigned __int8 **)(sub_986580(v19) + 48);
      v30 = v20;
      if ( v20 )
      {
        sub_B96E90((__int64)&v30, (__int64)v20, 1);
        if ( v30 )
        {
          v31[0] = v30;
          sub_B96E90((__int64)v31, (__int64)v30, 1);
          v21 = v31[0];
          *a1 = v31[0];
          if ( v21 )
          {
            sub_B96E90((__int64)a1, (__int64)v21, 1);
            v22 = v31[0];
            a1[1] = v31[0];
            if ( v22 )
            {
              sub_B96E90((__int64)(a1 + 1), (__int64)v22, 1);
              if ( v31[0] )
                sub_B91220((__int64)v31, (__int64)v31[0]);
            }
          }
          else
          {
            a1[1] = 0;
          }
          if ( v30 )
            sub_B91220((__int64)&v30, (__int64)v30);
          return a1;
        }
      }
    }
    v23 = **(_QWORD **)(a2 + 32);
    if ( !v23 )
    {
      *a1 = 0;
      a1[1] = 0;
      return a1;
    }
    v24 = *(unsigned __int8 **)(sub_986580(v23) + 48);
    v31[0] = v24;
    if ( v24 )
    {
      sub_B96E90((__int64)v31, (__int64)v24, 1);
      v25 = v31[0];
      *a1 = v31[0];
      if ( v25 )
      {
        sub_B96E90((__int64)a1, (__int64)v25, 1);
        v26 = v31[0];
        a1[1] = v31[0];
        if ( v26 )
        {
          sub_B96E90((__int64)(a1 + 1), (__int64)v26, 1);
          if ( v31[0] )
            sub_B91220((__int64)v31, (__int64)v31[0]);
        }
        return a1;
      }
    }
    else
    {
      *a1 = 0;
    }
    a1[1] = 0;
    return a1;
  }
  v15 = 0;
  do
  {
    if ( *(_BYTE *)*v12 == 6 )
    {
      if ( v15 )
      {
        sub_B10CB0(&v30, *v12);
        v31[0] = v29;
        if ( v29 )
        {
          sub_B96E90((__int64)v31, (__int64)v29, 1);
          v27 = v31[0];
          *a1 = v31[0];
          if ( v27 )
          {
            sub_B976B0((__int64)v31, v27, (__int64)a1);
            v31[0] = 0;
          }
        }
        else
        {
          *a1 = 0;
        }
        v28 = v30;
        a1[1] = v30;
        if ( v28 )
        {
          sub_B976B0((__int64)&v30, v28, (__int64)(a1 + 1));
          v30 = 0;
        }
        if ( v31[0] )
          sub_B91220((__int64)v31, (__int64)v31[0]);
        if ( v30 )
          sub_B91220((__int64)&v30, (__int64)v30);
        goto LABEL_18;
      }
      sub_B10CB0(v31, *v12);
      if ( v29 )
        sub_B91220((__int64)&v29, (__int64)v29);
      v15 = v31[0];
      v29 = v31[0];
      if ( v31[0] )
      {
        sub_B976B0((__int64)v31, v31[0], (__int64)&v29);
        v15 = v29;
      }
    }
    ++v12;
  }
  while ( v14 != v12 );
  if ( !v15 )
    goto LABEL_22;
  v31[0] = v15;
  sub_B96E90((__int64)v31, (__int64)v15, 1);
  v16 = v31[0];
  *a1 = v31[0];
  if ( v16 )
  {
    sub_B96E90((__int64)a1, (__int64)v16, 1);
    v17 = v31[0];
    a1[1] = v31[0];
    if ( v17 )
    {
      sub_B96E90((__int64)(a1 + 1), (__int64)v17, 1);
      if ( v31[0] )
        sub_B91220((__int64)v31, (__int64)v31[0]);
    }
  }
  else
  {
    a1[1] = 0;
  }
LABEL_18:
  if ( v29 )
    sub_B91220((__int64)&v29, (__int64)v29);
  return a1;
}
