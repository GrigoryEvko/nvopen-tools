// Function: sub_2D13500
// Address: 0x2d13500
//
void __fastcall sub_2D13500(__int64 **a1, __int64 a2, char a3, int a4)
{
  __int64 *v5; // rdx
  __int64 *v6; // r12
  __int64 *v9; // r15
  __int64 *v10; // r12
  __int64 *v11; // rsi
  __int64 *v12; // rdi
  char *v13; // r13
  __int64 *v14; // r14
  __int64 v15; // r15
  char *v16; // rsi
  char *v17; // r15
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  __int64 *v20; // r14
  __int64 v21; // r12
  __int64 *i; // r15
  __int64 v23; // rsi
  __int64 *v24; // r13
  __int64 *v26; // [rsp+8h] [rbp-68h]
  __int64 *v27; // [rsp+18h] [rbp-58h] BYREF
  char *v28; // [rsp+20h] [rbp-50h] BYREF
  char *v29; // [rsp+28h] [rbp-48h]
  char *v30; // [rsp+30h] [rbp-40h]

  v5 = *a1;
  v6 = a1[1];
  v28 = 0;
  v29 = 0;
  v30 = 0;
  if ( v5 != v6 )
  {
    v9 = v5;
    do
    {
      if ( !(unsigned __int8)sub_2D10740(*v9) && a2 == sub_2D10750(*v9) && (unsigned int)sub_2D109A0(*v9) == a4 )
      {
        v16 = v29;
        if ( v29 == v30 )
        {
          sub_2D10A40((__int64)&v28, v29, v9);
        }
        else
        {
          if ( v29 )
          {
            *(_QWORD *)v29 = *v9;
            v16 = v29;
          }
          v29 = v16 + 8;
        }
      }
      ++v9;
    }
    while ( v6 != v9 );
    v10 = (__int64 *)v28;
    v26 = (__int64 *)v29;
    if ( a3 )
    {
      if ( v28 == v29 )
      {
        v27 = 0;
LABEL_22:
        if ( v26 )
          j_j___libc_free_0((unsigned __int64)v26);
        return;
      }
      v17 = v29;
      v18 = v29 - v28;
      _BitScanReverse64(&v19, (v29 - v28) >> 3);
      sub_2D132C0(
        v28,
        v29,
        2LL * (int)(63 - (v19 ^ 0x3F)),
        (unsigned __int8 (__fastcall *)(__int64, __int64))sub_2D108F0);
      if ( v18 <= 128 )
      {
        sub_2D12E80(v10, v26, (__int64 (__fastcall *)(__int64, __int64))sub_2D108F0);
      }
      else
      {
        v20 = v10 + 16;
        sub_2D12E80(v10, v10 + 16, (__int64 (__fastcall *)(__int64, __int64))sub_2D108F0);
        if ( v10 + 16 != (__int64 *)v17 )
        {
          do
          {
            v21 = *v20;
            for ( i = v20; ; i[1] = *i )
            {
              v23 = *(i - 1);
              v24 = i--;
              if ( !sub_2D108F0(v21, v23) )
                break;
            }
            *v24 = v21;
            ++v20;
          }
          while ( v26 != v20 );
        }
      }
      v10 = (__int64 *)v28;
      v26 = (__int64 *)v29;
    }
    v27 = 0;
    if ( v26 != v10 )
    {
      do
      {
        v27 = (__int64 *)*v10;
        if ( !(unsigned __int8)sub_2D10740((__int64)v27) )
        {
          v11 = a1[35];
          if ( v11 == a1[36] )
          {
            sub_2D10A40((__int64)(a1 + 34), v11, &v27);
            v12 = v27;
          }
          else
          {
            v12 = v27;
            if ( v11 )
            {
              *v11 = (__int64)v27;
              v11 = a1[35];
            }
            a1[35] = v11 + 1;
          }
          sub_2D10730((__int64)v12);
          v13 = v29;
          if ( v29 != v28 )
          {
            v14 = (__int64 *)v28;
            do
            {
              v15 = *v14;
              if ( !(unsigned __int8)sub_2D10740(*v14) && !(unsigned __int8)sub_2D10760(v27, v15) )
              {
                sub_2D11630((__int64)v27, v15);
                sub_2D10720(v15, (__int64)v27);
                sub_2D10730(v15);
              }
              ++v14;
            }
            while ( v13 != (char *)v14 );
          }
        }
        ++v10;
      }
      while ( v26 != v10 );
      v26 = (__int64 *)v28;
    }
    goto LABEL_22;
  }
}
