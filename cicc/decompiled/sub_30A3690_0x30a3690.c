// Function: sub_30A3690
// Address: 0x30a3690
//
__int64 __fastcall sub_30A3690(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r12
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  const char *v8; // rax
  size_t v9; // rdx
  void *v10; // rdi
  unsigned __int8 *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // r13d
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  _WORD *v18; // rdx
  unsigned __int8 **v19; // r12
  unsigned __int8 *v20; // r9
  unsigned __int8 **i; // r14
  __int64 v22; // rdi
  _WORD *v23; // rdx
  __int64 v24; // rdi
  _BYTE *v25; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // r14
  unsigned int v29; // r12d
  __int64 v30; // rdi
  __m128i *v31; // rdx
  __m128i v32; // xmm0
  void *v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  int v37; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v38; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v39; // [rsp+18h] [rbp-A8h]
  size_t v40; // [rsp+18h] [rbp-A8h]
  _QWORD v41[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+30h] [rbp-90h]
  __int64 v43; // [rsp+38h] [rbp-88h]
  __int64 v44; // [rsp+40h] [rbp-80h]
  unsigned __int64 v45; // [rsp+48h] [rbp-78h]
  __int64 v46; // [rsp+50h] [rbp-70h]
  __int64 v47; // [rsp+58h] [rbp-68h]
  unsigned __int8 **v48; // [rsp+60h] [rbp-60h]
  unsigned __int8 **v49; // [rsp+68h] [rbp-58h]
  __int64 v50; // [rsp+70h] [rbp-50h]
  unsigned __int64 v51; // [rsp+78h] [rbp-48h]
  __int64 v52; // [rsp+80h] [rbp-40h]
  __int64 v53; // [rsp+88h] [rbp-38h]

  v5 = *a2;
  v6 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v6 <= 0x11u )
  {
    v5 = sub_CB6200(v5, "SCCs for Function ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44CB490);
    v6[1].m128i_i16[0] = 8302;
    *v6 = si128;
    *(_QWORD *)(v5 + 32) += 18LL;
  }
  v8 = sub_BD5D20(a3);
  v10 = *(void **)(v5 + 32);
  v11 = (unsigned __int8 *)v8;
  v12 = *(_QWORD *)(v5 + 24) - (_QWORD)v10;
  if ( v12 < v9 )
  {
    v35 = sub_CB6200(v5, v11, v9);
    v10 = *(void **)(v35 + 32);
    v5 = v35;
    v12 = *(_QWORD *)(v35 + 24) - (_QWORD)v10;
  }
  else if ( v9 )
  {
    v40 = v9;
    memcpy(v10, v11, v9);
    v33 = (void *)(*(_QWORD *)(v5 + 32) + v40);
    v34 = *(_QWORD *)(v5 + 24) - (_QWORD)v33;
    *(_QWORD *)(v5 + 32) = v33;
    v10 = v33;
    if ( v34 > 0xD )
      goto LABEL_6;
    goto LABEL_44;
  }
  if ( v12 > 0xD )
  {
LABEL_6:
    qmemcpy(v10, " in PostOrder:", 14);
    *(_QWORD *)(v5 + 32) += 14LL;
    goto LABEL_7;
  }
LABEL_44:
  sub_CB6200(v5, " in PostOrder:", 0xEu);
LABEL_7:
  v13 = *(_QWORD *)(a3 + 80);
  v41[0] = 0;
  v41[1] = 0;
  v42 = 0;
  if ( v13 )
    v13 -= 24;
  v44 = 0;
  v14 = 0;
  v43 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  sub_2572230((__int64)v41, v13);
  sub_25725D0((__int64)v41);
  while ( v49 != v48 )
  {
    v15 = *a2;
    v16 = *(_QWORD *)(*a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v16) <= 5 )
    {
      v15 = sub_CB6200(v15, "\nSCC #", 6u);
    }
    else
    {
      *(_DWORD *)v16 = 1128485642;
      *(_WORD *)(v16 + 4) = 8992;
      *(_QWORD *)(v15 + 32) += 6LL;
    }
    v17 = sub_CB59D0(v15, (unsigned int)++v14);
    v18 = *(_WORD **)(v17 + 32);
    if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 1u )
    {
      sub_CB6200(v17, (unsigned __int8 *)": ", 2u);
    }
    else
    {
      *v18 = 8250;
      *(_QWORD *)(v17 + 32) += 2LL;
    }
    v19 = v49;
    if ( v49 != v48 )
    {
      v20 = *v48;
      for ( i = v48 + 1; ; ++i )
      {
        sub_A5BF40(v20, *a2, 0, 0);
        if ( v19 == i )
          break;
        v22 = *a2;
        v20 = *i;
        v23 = *(_WORD **)(*a2 + 32);
        if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v23 <= 1u )
        {
          v39 = *i;
          sub_CB6200(v22, (unsigned __int8 *)", ", 2u);
          v20 = v39;
        }
        else
        {
          *v23 = 8236;
          *(_QWORD *)(v22 + 32) += 2LL;
        }
      }
      if ( (char *)v49 - (char *)v48 == 8 )
      {
        v38 = *v48;
        v27 = *((_QWORD *)*v48 + 6) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (unsigned __int8 *)v27 != *v48 + 48 )
        {
          if ( !v27 )
            BUG();
          v28 = v27 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v27 - 24) - 30 <= 0xA )
          {
            v37 = sub_B46E30(v28);
            if ( v37 )
            {
              v29 = 0;
              while ( v38 != (unsigned __int8 *)sub_B46EC0(v28, v29) )
              {
                if ( v37 == ++v29 )
                  goto LABEL_21;
              }
              v30 = *a2;
              v31 = *(__m128i **)(*a2 + 32);
              if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v31 <= 0x10u )
              {
                sub_CB6200(v30, " (Has self-loop).", 0x11u);
              }
              else
              {
                v32 = _mm_load_si128((const __m128i *)&xmmword_3F70860);
                v31[1].m128i_i8[0] = 46;
                *v31 = v32;
                *(_QWORD *)(v30 + 32) += 17LL;
              }
            }
          }
        }
      }
    }
LABEL_21:
    sub_25725D0((__int64)v41);
  }
  if ( v51 )
    j_j___libc_free_0(v51);
  if ( v48 )
    j_j___libc_free_0((unsigned __int64)v48);
  if ( v45 )
    j_j___libc_free_0(v45);
  sub_C7D6A0(v42, 16LL * (unsigned int)v44, 8);
  v24 = *a2;
  v25 = *(_BYTE **)(*a2 + 32);
  if ( *(_BYTE **)(*a2 + 24) == v25 )
  {
    sub_CB6200(v24, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v25 = 10;
    ++*(_QWORD *)(v24 + 32);
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
