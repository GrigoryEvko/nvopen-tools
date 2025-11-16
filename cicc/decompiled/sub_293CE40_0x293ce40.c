// Function: sub_293CE40
// Address: 0x293ce40
//
__m128i *__fastcall sub_293CE40(__m128i *a1, _QWORD *a2, __int64 a3, unsigned __int64 a4, const __m128i *a5)
{
  unsigned __int64 v9; // rdx
  __int64 v10; // r13
  unsigned __int64 *v11; // rcx
  unsigned __int64 *v12; // rax
  unsigned __int64 *v13; // rsi
  __int64 v15; // r13
  __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned __int64 *v20; // rax
  unsigned __int64 *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // r10
  __int64 v25; // r11
  __int64 v26; // rax
  __int16 v27; // dx
  __int16 v28; // cx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rsi
  __int16 v32; // dx
  __int16 v33; // ax
  char v34; // r8
  char v35; // al
  __int16 v36; // cx
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v39; // [rsp+8h] [rbp-78h]
  const __m128i *v40; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v41; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v42; // [rsp+48h] [rbp-38h]

  if ( *(_BYTE *)a4 == 22 )
  {
    v9 = a5[1].m128i_u64[0];
    v10 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 80LL);
    v41 = a4;
    v11 = a2 + 1;
    v42 = v9;
    if ( v10 )
      v10 -= 24;
    v12 = (unsigned __int64 *)a2[2];
    v13 = a2 + 1;
    if ( !v12 )
      goto LABEL_5;
    do
    {
      while ( a4 <= v12[4] && (a4 != v12[4] || v9 <= v12[5]) )
      {
        v13 = v12;
        v12 = (unsigned __int64 *)v12[2];
        if ( !v12 )
          goto LABEL_13;
      }
      v12 = (unsigned __int64 *)v12[3];
    }
    while ( v12 );
LABEL_13:
    if ( v11 == v13 || a4 < v13[4] || a4 == v13[4] && v9 < v13[5] )
    {
LABEL_5:
      v40 = (const __m128i *)&v41;
      v13 = sub_293C9D0(a2, v13, &v40);
    }
    sub_293A710(a1, v10, *(_QWORD *)(v10 + 56), 1, a4, a5, (__int64)(v13 + 6));
  }
  else if ( *(_BYTE *)a4 <= 0x1Cu )
  {
    sub_293A710(a1, *(_QWORD *)(a3 + 40), a3 + 24, 0, a4, a5, 0);
  }
  else
  {
    v15 = *(_QWORD *)(a4 + 40);
    v16 = a2[139];
    if ( v15 )
    {
      v17 = (unsigned int)(*(_DWORD *)(v15 + 44) + 1);
      v18 = *(_DWORD *)(v15 + 44) + 1;
    }
    else
    {
      v17 = 0;
      v18 = 0;
    }
    if ( v18 < *(_DWORD *)(v16 + 32) && *(_QWORD *)(*(_QWORD *)(v16 + 24) + 8 * v17) )
    {
      v19 = a5[1].m128i_u64[0];
      v20 = (unsigned __int64 *)a2[2];
      v41 = a4;
      v21 = a2 + 1;
      v42 = v19;
      if ( !v20 )
        goto LABEL_41;
      do
      {
        while ( a4 <= v20[4] && (a4 != v20[4] || v19 <= v20[5]) )
        {
          v21 = v20;
          v20 = (unsigned __int64 *)v20[2];
          if ( !v20 )
            goto LABEL_30;
        }
        v20 = (unsigned __int64 *)v20[3];
      }
      while ( v20 );
LABEL_30:
      if ( a2 + 1 == v21 || a4 < v21[4] || a4 == v21[4] && v19 < v21[5] )
      {
LABEL_41:
        v40 = (const __m128i *)&v41;
        v21 = sub_293C9D0(a2, v21, &v40);
      }
      v22 = *(_QWORD *)(a4 + 32);
      v23 = (__int64)(v21 + 6);
      v24 = 0;
      if ( !v22 )
        BUG();
      v25 = *(_QWORD *)(v22 + 16);
      if ( *(_BYTE *)(v22 - 24) == 84 )
      {
        v39 = *(_QWORD *)(v22 + 16);
        v30 = sub_AA5190(v39);
        v25 = v39;
        v24 = 0;
        v31 = v30;
        v33 = v32;
        v22 = v31;
        if ( v31 )
        {
          v34 = v33;
          v35 = HIBYTE(v33);
        }
        else
        {
          v35 = 0;
          v34 = 0;
        }
        LOBYTE(v36) = v34;
        HIBYTE(v36) = v35;
        LOWORD(v24) = v36;
      }
      if ( v22 != v25 + 48 )
      {
        v37 = v24;
        v26 = sub_AA5FF0(v22);
        v24 = v37;
        v28 = v27;
        v22 = v26;
        LOWORD(v24) = v28;
      }
      sub_293A710(a1, v15, v22, v24, a4, a5, v23);
    }
    else
    {
      v29 = sub_ACADE0(*(__int64 ***)(a4 + 8));
      sub_293A710(a1, *(_QWORD *)(a3 + 40), a3 + 24, 0, v29, a5, 0);
    }
  }
  return a1;
}
