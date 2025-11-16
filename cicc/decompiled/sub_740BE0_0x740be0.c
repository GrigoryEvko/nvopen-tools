// Function: sub_740BE0
// Address: 0x740be0
//
_BYTE *__fastcall sub_740BE0(
        __int64 a1,
        int a2,
        int a3,
        __int64 a4,
        int a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        const __m128i *a9,
        int a10,
        _DWORD *a11,
        __int64 a12)
{
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r12
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdi
  const __m128i *v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  _UNKNOWN *__ptr32 *v33; // r8
  unsigned __int8 v34; // [rsp+Fh] [rbp-61h]
  unsigned int v35; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v36[10]; // [rsp+20h] [rbp-50h] BYREF

  v35 = 0;
  v15 = sub_730C30((__int64 *)a1, a2, a3, a8, a9, a10);
  if ( v15 )
  {
    v18 = v15;
    v19 = *(unsigned __int8 *)(v15 + 80);
    if ( (_BYTE)v19 == 16 )
    {
      v18 = **(_QWORD **)(v18 + 88);
      v19 = *(unsigned __int8 *)(v18 + 80);
      if ( (_BYTE)v19 != 24 )
      {
LABEL_4:
        LOBYTE(v17) = a4 != 0;
        if ( (_BYTE)v19 != 2 )
        {
LABEL_5:
          if ( (_BYTE)v19 == 9 )
          {
            v34 = v17;
            v27 = *(_QWORD *)(v18 + 88);
            if ( !a5 )
            {
              v28 = sub_6EA380(v27, 1, 1, 0);
              v17 = v34;
              a12 = v28;
              if ( !v28 )
                goto LABEL_12;
              v21 = v28;
              if ( !v34 )
              {
                v21 = v28;
                goto LABEL_22;
              }
              goto LABEL_20;
            }
            sub_72D510(v27, a12, 1);
          }
          else
          {
            if ( (_BYTE)v19 != 8 )
            {
              if ( (unsigned __int8)v19 > 0x14u )
                goto LABEL_12;
              v20 = 1180672;
              if ( !_bittest64(&v20, v19) || !a4 || !a5 && !(unsigned int)sub_8D2E30(a4) )
                goto LABEL_12;
              if ( (unsigned int)sub_8DBE70(a4) || (unsigned int)sub_89A370(a7) )
              {
                v25 = *(_QWORD *)(sub_7D0010(v18, *(_BYTE *)(a1 + 177) & 1) + 88);
                if ( a6 )
                {
                  v36[0] = (__int64)sub_724DC0();
                  sub_724C70(v36[0], 12);
                  sub_7249B0(v36[0], 11);
                  v30 = (const __m128i *)v36[0];
                  *(_QWORD *)(v36[0] + 192) = a7;
                  v30[11].m128i_i64[1] = v25;
                  v30[8].m128i_i64[0] = dword_4D03B80;
                  v25 = sub_73A460(v30, 11, v31, v32, v33);
                  sub_724E30((__int64)v36);
                }
                v26 = v25;
                v21 = 0;
                sub_70FDD0(v26, a12, a4, 0);
              }
              else
              {
                v29 = v18;
                v21 = 0;
                sub_82D750(v29, a6, a7, a4, a12, &v35);
              }
LABEL_23:
              if ( !v35 )
                return (_BYTE *)v21;
              goto LABEL_14;
            }
            if ( !a5 )
              goto LABEL_12;
            v34 = v17;
            sub_73F1E0(*(__m128i ***)(v18 + 88), a12);
            v16 = v35;
          }
          v17 = v34;
          if ( !v34 )
          {
            v21 = 0;
            goto LABEL_22;
          }
          v21 = 0;
LABEL_20:
          v24 = *(_QWORD *)(a12 + 128);
          if ( a4 != v24
            && !(unsigned int)sub_8D97D0(a4, *(_QWORD *)(a12 + 128), 0, v16, v17)
            && (!(unsigned int)sub_8E1010(v24, 1, 0, 0, 0, a12, a4, 0, 0, 1, 0, (__int64)v36, 0)
             || !(unsigned int)sub_8DD690(v36, v24, 1, a12, a4, 0)) )
          {
            v35 = 1;
          }
LABEL_22:
          if ( a6 )
            goto LABEL_13;
          goto LABEL_23;
        }
LABEL_16:
        v21 = *(_QWORD *)(v18 + 88);
        if ( !(_BYTE)v17 )
          goto LABEL_22;
        v23 = v21;
        if ( !v21 )
          v23 = a12;
        a12 = v23;
        goto LABEL_20;
      }
    }
    else if ( (_BYTE)v19 != 24 )
    {
      goto LABEL_4;
    }
    v18 = *(_QWORD *)(v18 + 88);
    if ( !v18 )
      goto LABEL_12;
    v19 = *(unsigned __int8 *)(v18 + 80);
    LOBYTE(v17) = a4 != 0;
    if ( (_BYTE)v19 != 2 )
      goto LABEL_5;
    goto LABEL_16;
  }
LABEL_12:
  v35 = 1;
  if ( a6 )
LABEL_13:
    v35 = 1;
LABEL_14:
  *a11 = 1;
  return sub_72C9A0();
}
