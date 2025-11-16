// Function: sub_22CF7C0
// Address: 0x22cf7c0
//
__int64 __fastcall sub_22CF7C0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v9; // r8
  __int64 v10; // r14
  unsigned __int64 v11; // rsi
  __int64 v12; // r8
  __int64 *v13; // r9
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned __int8 v18; // al
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rdx
  bool v24; // al
  unsigned __int8 *v25; // rax
  char v26; // al
  _QWORD **v27; // rax
  int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // r12
  __int64 v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-C8h]
  __int64 v41; // [rsp+10h] [rbp-C0h]
  __int64 v42; // [rsp+18h] [rbp-B8h]
  __int64 v43; // [rsp+28h] [rbp-A8h]
  __int64 v44; // [rsp+28h] [rbp-A8h]
  __int64 v45; // [rsp+28h] [rbp-A8h]
  __int64 v46; // [rsp+28h] [rbp-A8h]
  __int64 v47; // [rsp+28h] [rbp-A8h]
  __int64 v49; // [rsp+30h] [rbp-A0h]
  unsigned int v50; // [rsp+30h] [rbp-A0h]
  __int64 v51; // [rsp+30h] [rbp-A0h]
  __int64 v53; // [rsp+48h] [rbp-88h]
  __m128i v54; // [rsp+50h] [rbp-80h] BYREF
  __int64 v55; // [rsp+60h] [rbp-70h]
  __int64 v56; // [rsp+68h] [rbp-68h]
  __int64 v57; // [rsp+70h] [rbp-60h]
  __int64 v58; // [rsp+78h] [rbp-58h]
  __int64 v59; // [rsp+80h] [rbp-50h]
  __int64 v60; // [rsp+88h] [rbp-48h]
  __int16 v61; // [rsp+90h] [rbp-40h]

  v9 = sub_B43CA0(a5);
  v10 = v9 + 312;
  if ( *(_BYTE *)(*(_QWORD *)(a3 + 8) + 8LL) != 14 )
    goto LABEL_2;
  v45 = v9;
  v24 = sub_AC30F0(a4);
  v9 = v45;
  if ( !v24 )
    goto LABEL_2;
  v54 = (__m128i)(unsigned __int64)v10;
  v61 = 257;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v25 = sub_BD3E50((unsigned __int8 *)a3, (__int64)&v54);
  v26 = sub_9B6260((__int64)v25, &v54, 0);
  v9 = v45;
  if ( !v26 )
    goto LABEL_2;
  v27 = *(_QWORD ***)(a4 + 8);
  v28 = *((unsigned __int8 *)v27 + 8);
  if ( (unsigned int)(v28 - 17) > 1 )
  {
    v30 = sub_BCB2A0(*v27);
  }
  else
  {
    BYTE4(v53) = (_BYTE)v28 == 18;
    LODWORD(v53) = *((_DWORD *)v27 + 8);
    v29 = (__int64 *)sub_BCB2A0(*v27);
    v30 = sub_BCE1B0(v29, v53);
  }
  v9 = v45;
  if ( a2 != 32 )
  {
    if ( a2 == 33 )
      return sub_AD6400(v30);
LABEL_2:
    v11 = sub_22C1480(a1, v9);
    if ( a6 )
    {
      sub_22CDEF0((__int64)&v54, v11, a3, *(_QWORD *)(a5 + 40), a5);
      if ( v54.m128i_i8[0] != 2 )
      {
LABEL_4:
        v42 = sub_22BE9F0(a2, a4, (__int64)&v54, v10);
        if ( v42 )
          goto LABEL_5;
        goto LABEL_8;
      }
    }
    else
    {
      sub_22CCDF0((__int64)&v54, v11, (unsigned __int8 *)a3, a5, v12, v13);
      if ( v54.m128i_i8[0] != 2 )
        goto LABEL_4;
    }
    v42 = sub_9719A0(a2, (_BYTE *)v54.m128i_i64[1], a4, v10, 0, 0);
    if ( v42 )
      goto LABEL_5;
LABEL_8:
    v15 = *(_QWORD *)(a5 + 40);
    v16 = *(_QWORD *)(v15 + 16);
    if ( v16 )
    {
      while ( 1 )
      {
        v17 = *(_QWORD *)(v16 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
          break;
        v16 = *(_QWORD *)(v16 + 8);
        if ( !v16 )
          goto LABEL_5;
      }
      v18 = *(_BYTE *)a3;
      if ( *(_BYTE *)a3 == 84 )
      {
        if ( *(_QWORD *)(a3 + 40) == v15 )
        {
          v50 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
          if ( v50 )
          {
            v46 = *(_QWORD *)(a5 + 40);
            v31 = sub_22CF6C0(
                    a1,
                    a2,
                    **(_QWORD **)(a3 - 8),
                    a4,
                    *(_QWORD *)(*(_QWORD *)(a3 - 8) + 32LL * *(unsigned int *)(a3 + 72)),
                    v15,
                    a5);
            v40 = v16;
            v15 = v46;
            v47 = a4;
            v32 = 8;
            v33 = a3;
            v34 = v31;
            v41 = 8LL * v50;
            while ( 1 )
            {
              if ( !v34 )
              {
LABEL_36:
                a3 = v33;
                v16 = v40;
                a4 = v47;
                v18 = *(_BYTE *)a3;
                goto LABEL_11;
              }
              if ( v41 == v32 )
                break;
              v35 = *(_QWORD *)(v33 - 8);
              v51 = v15;
              v36 = v35 + 32LL * *(unsigned int *)(v33 + 72);
              v37 = *(_QWORD *)(v35 + 4 * v32);
              v38 = *(_QWORD *)(v36 + v32);
              v32 += 8;
              v39 = sub_22CF6C0(a1, a2, v37, v47, v38, v15, a5);
              v15 = v51;
              if ( v34 != v39 )
                goto LABEL_36;
            }
            v42 = v34;
          }
          goto LABEL_5;
        }
      }
      else
      {
LABEL_11:
        if ( v18 > 0x1Cu && v15 == *(_QWORD *)(a3 + 40) )
          goto LABEL_5;
        v17 = *(_QWORD *)(v16 + 24);
      }
      v43 = v15;
      v19 = sub_22CF6C0(a1, a2, a3, a4, *(_QWORD *)(v17 + 40), v15, a5);
      v20 = v43;
      v49 = v19;
      if ( v19 )
      {
        v44 = a3;
        v21 = v16;
        v22 = v20;
        while ( 1 )
        {
          v21 = *(_QWORD *)(v21 + 8);
          if ( !v21 )
            break;
          while ( 1 )
          {
            v23 = *(_QWORD *)(v21 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v23 - 30) > 0xAu )
              break;
            if ( v49 != sub_22CF6C0(a1, a2, v44, a4, *(_QWORD *)(v23 + 40), v22, a5) )
              goto LABEL_5;
            v21 = *(_QWORD *)(v21 + 8);
            if ( !v21 )
              goto LABEL_20;
          }
        }
LABEL_20:
        v42 = v49;
      }
    }
LABEL_5:
    sub_22C0090((unsigned __int8 *)&v54);
    return v42;
  }
  return sub_AD6450(v30);
}
