// Function: sub_21E39F0
// Address: 0x21e39f0
//
void __fastcall sub_21E39F0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // rdx
  __int64 v11; // r12
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rcx
  __int64 v15; // rsi
  int v16; // edx
  int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int16 v23; // r14
  __int16 v24; // r14
  __int64 v25; // rsi
  _QWORD *v26; // r15
  __int64 v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // [rsp-48h] [rbp-48h] BYREF
  int v30; // [rsp-40h] [rbp-40h]

  v27 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 88LL);
  v28 = *(_QWORD **)(v27 + 24);
  if ( *(_DWORD *)(v27 + 32) > 0x40u )
    v28 = (_QWORD *)*v28;
  if ( (_DWORD)v28 == 4005 )
  {
    sub_21DF3A0(a1, a2, (__int64)v28, a7, a8, a9);
    return;
  }
  if ( (unsigned int)v28 <= 0xFA5 )
  {
    if ( (_DWORD)v28 == 3659 )
    {
      sub_21DF050(a1, a2);
      return;
    }
    if ( (unsigned int)v28 <= 0xE4B )
    {
      if ( (unsigned int)v28 > 0xE3B )
      {
        if ( (_DWORD)v28 != 3658 )
          return;
        goto LABEL_16;
      }
      if ( (unsigned int)v28 <= 0xE39 )
        return;
    }
    else
    {
      if ( (unsigned int)v28 > 0xF5A )
      {
        if ( (_DWORD)v28 == 4004 )
        {
          v9 = *(_QWORD *)(a1 + 32);
          if ( *(_DWORD *)(v9 + 252) > 0x3Cu && *(_DWORD *)(v9 + 248) > 0x31u && *(_DWORD *)(a1 - 144) )
          {
            v11 = *(_QWORD *)(a2 + 32);
            v12 = *(_QWORD *)(v11 + 80);
            v13 = *(unsigned __int16 *)(v12 + 24);
            if ( v13 != 10 && v13 != 32 )
              v12 = 0;
            v14 = *(_QWORD *)(v11 + 160);
            v15 = *(_QWORD *)(v11 + 200);
            v16 = *(unsigned __int16 *)(v14 + 24);
            if ( v16 == 10 || v16 == 32 )
            {
              v17 = *(unsigned __int16 *)(v15 + 24);
              if ( v17 == 10 || v17 == 32 )
              {
                if ( v12 )
                {
                  v18 = *(_QWORD *)(v12 + 88);
                  if ( *(_DWORD *)(v18 + 32) == 1 )
                  {
                    v19 = *(_QWORD *)(v14 + 88);
                    if ( *(_DWORD *)(v19 + 32) == 1 )
                    {
                      v20 = *(_QWORD *)(v15 + 88);
                      if ( *(_DWORD *)(v20 + 32) == 1 )
                      {
                        v21 = *(_QWORD *)(v18 + 24);
                        v22 = *(_QWORD *)(v19 + 24);
                        if ( *(_QWORD *)(v20 + 24) )
                        {
                          v23 = v22 == 1;
                          if ( v21 == 1 )
                            v24 = v23 + 383;
                          else
                            v24 = v23 + 381;
                        }
                        else if ( v21 == 1 )
                        {
                          v24 = (v22 == 1) + 387;
                        }
                        else
                        {
                          v24 = (v22 == 1) + 385;
                        }
                        v25 = *(_QWORD *)(a2 + 72);
                        v26 = *(_QWORD **)(a1 - 176);
                        v29 = v25;
                        if ( v25 )
                          sub_1623A60((__int64)&v29, v25, 2);
                        v30 = *(_DWORD *)(a2 + 64);
                        sub_1D2CD40(
                          v26,
                          v24,
                          (__int64)&v29,
                          5,
                          0,
                          a9,
                          *(_OWORD *)(v11 + 40),
                          *(_OWORD *)(v11 + 120),
                          *(_OWORD *)(v11 + 240));
                        if ( v29 )
                          sub_161E7C0((__int64)&v29, v29);
                      }
                    }
                  }
                }
              }
            }
          }
          else
          {
            nullsub_2021();
          }
        }
        return;
      }
      if ( (unsigned int)v28 <= 0xF58 )
        return;
    }
LABEL_29:
    sub_21E3600(a1, a2, a3, a4, a5);
    return;
  }
  if ( (_DWORD)v28 == 4193 )
    goto LABEL_29;
  if ( (unsigned int)v28 <= 0x1061 )
  {
    if ( (_DWORD)v28 == 4110 )
    {
      sub_21DF5E0(a1, a2, (__int64)v28, a7, a8, a9);
      return;
    }
    if ( (_DWORD)v28 == 4190 )
      goto LABEL_29;
    if ( (_DWORD)v28 != 4109 )
      return;
LABEL_16:
    sub_21DF9F0(a1, a2);
    return;
  }
  if ( (unsigned int)v28 > 0x11A1 )
  {
    if ( (_DWORD)v28 != 4516 )
      return;
    goto LABEL_16;
  }
  if ( (unsigned int)v28 > 0x119F )
  {
    sub_21DF6A0(a1, a2, a3, a4, a5);
    return;
  }
  if ( (_DWORD)v28 == 4208 )
    goto LABEL_16;
}
