// Function: sub_3866590
// Address: 0x3866590
//
__int64 __fastcall sub_3866590(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6, __m128i a7)
{
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 v10; // r12
  _QWORD *v11; // rbx
  int v12; // eax
  __int64 v13; // rdi
  unsigned __int64 v14; // r14
  int v15; // r9d
  unsigned int v16; // ecx
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  int v19; // r8d
  int v20; // r9d
  int v21; // eax
  __int64 v22; // rax
  _BYTE *v23; // rdi
  __int64 v24; // r9
  unsigned int v25; // r11d
  __int64 v26; // r10
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rax
  int v30; // edi
  __int64 *v31; // r12
  __int64 v32; // rdx
  unsigned int v33; // ecx
  __int64 v34; // rdx
  unsigned int v35; // ecx
  __int64 result; // rax
  _QWORD *v37; // r12
  unsigned int v38; // r14d
  _QWORD *v39; // rbx
  __int64 v40; // rax
  __int64 *v41; // r15
  __int64 v42; // [rsp+0h] [rbp-E0h]
  char v43; // [rsp+9h] [rbp-D7h]
  unsigned __int8 v44; // [rsp+Ah] [rbp-D6h]
  char v45; // [rsp+Bh] [rbp-D5h]
  unsigned int v46; // [rsp+Ch] [rbp-D4h]
  __int64 v47; // [rsp+18h] [rbp-C8h]
  char v48; // [rsp+23h] [rbp-BDh]
  int v49; // [rsp+24h] [rbp-BCh]
  int v50; // [rsp+28h] [rbp-B8h]
  __int64 *v51; // [rsp+28h] [rbp-B8h]
  int v54; // [rsp+4Ch] [rbp-94h]
  int v55; // [rsp+5Ch] [rbp-84h] BYREF
  __int64 v56; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v57; // [rsp+68h] [rbp-78h]
  __int64 v58; // [rsp+70h] [rbp-70h]
  int v59; // [rsp+78h] [rbp-68h]
  _BYTE *v60; // [rsp+80h] [rbp-60h] BYREF
  __int64 v61; // [rsp+88h] [rbp-58h]
  _BYTE v62[80]; // [rsp+90h] [rbp-50h] BYREF

  v7 = a1;
  v8 = a1 + 328;
  v45 = a5;
  v46 = *(_DWORD *)(v8 - 248);
  v42 = v8;
  v47 = *(_QWORD *)(v8 + 8);
  if ( v47 == v8 )
  {
    v25 = *(_DWORD *)(a2 + 16);
    if ( v25 )
    {
      v43 = 1;
      v44 = 0;
      goto LABEL_27;
    }
    v44 = 0;
    goto LABEL_69;
  }
  v10 = v7;
  v44 = 0;
  v43 = 1;
  v54 = 1;
  do
  {
    v58 = 0;
    v59 = 0;
    v60 = v62;
    v61 = 0x400000000LL;
    v55 = 1;
    v11 = *(_QWORD **)(v47 + 16);
    v56 = 0;
    v57 = 0;
    if ( v11 )
    {
      v48 = 1;
      v49 = 0;
      v50 = 0;
      do
      {
        while ( 1 )
        {
          v21 = *(_DWORD *)(v10 + 24);
          v18 = *v11 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v21 )
          {
            v12 = v21 - 1;
            v13 = *(_QWORD *)(v10 + 8);
            v14 = v18 | 4;
            v15 = 1;
            v16 = v12 & ((v18 | 4) ^ (v18 >> 9));
            v17 = *(_QWORD *)(v13 + 8LL * v16);
            if ( (v18 | 4) == v17 )
            {
LABEL_6:
              ++v50;
              v18 |= 4u;
            }
            else
            {
              while ( v17 != -4 )
              {
                v16 = v12 & (v15 + v16);
                v17 = *(_QWORD *)(v13 + 8LL * v16);
                if ( v14 == v17 )
                  goto LABEL_6;
                ++v15;
              }
              ++v49;
              v14 = *v11 & 0xFFFFFFFFFFFFFFFBLL;
            }
          }
          else
          {
            ++v49;
            v14 = *v11 & 0xFFFFFFFFFFFFFFFBLL;
          }
          if ( !(unsigned __int8)sub_3864EE0(v10, a2, v18, a4, (__int64)&v56, a3, a6, a7, &v55, v54, v45, 0) )
            break;
          v11 = (_QWORD *)v11[2];
          if ( !v11 )
            goto LABEL_18;
        }
        v22 = (unsigned int)v61;
        if ( (unsigned int)v61 >= HIDWORD(v61) )
        {
          sub_16CD150((__int64)&v60, v62, 0, 8, v19, v20);
          v22 = (unsigned int)v61;
        }
        v48 = 0;
        *(_QWORD *)&v60[8 * v22] = v14;
        v11 = (_QWORD *)v11[2];
        LODWORD(v61) = v61 + 1;
      }
      while ( v11 );
LABEL_18:
      v23 = v60;
      if ( !v46 || v48 != 1 )
      {
        if ( v50 > 1 || v49 > 0 && (v50 & 1) != 0 )
        {
          if ( v48 != 1 )
          {
            if ( &v60[8 * (unsigned int)v61] != v60 )
            {
              v51 = (__int64 *)&v60[8 * (unsigned int)v61];
              v41 = (__int64 *)v60;
              do
              {
                if ( !(unsigned __int8)sub_3864EE0(v10, a2, *v41, a4, (__int64)&v56, a3, a6, a7, &v55, v54, v45, 1) )
                {
                  v43 = 0;
                  v23 = v60;
                  goto LABEL_54;
                }
                ++v41;
              }
              while ( v51 != v41 );
              v23 = v60;
            }
LABEL_54:
            v44 = v48 ^ 1;
            goto LABEL_23;
          }
LABEL_56:
          v44 = 1;
          v43 &= v48;
          goto LABEL_23;
        }
        v43 &= v48;
      }
      else
      {
        if ( v55 == 2 )
          goto LABEL_23;
        if ( v50 > 1 )
        {
          v44 = 1;
          goto LABEL_23;
        }
        v48 = v50 & (v49 > 0);
        if ( v48 )
          goto LABEL_56;
        v43 &= 1u;
      }
    }
    else
    {
      if ( !v46 )
      {
        ++v54;
        goto LABEL_25;
      }
      v23 = v62;
      v43 &= 1u;
    }
LABEL_23:
    ++v54;
    if ( v23 != v62 )
      _libc_free((unsigned __int64)v23);
LABEL_25:
    j___libc_free_0(v57);
    v47 = *(_QWORD *)(v47 + 8);
  }
  while ( v42 != v47 );
  v25 = *(_DWORD *)(a2 + 16);
  v7 = v10;
  if ( v25 )
  {
LABEL_27:
    v24 = 1;
    v26 = 0;
    if ( v25 > 1 )
    {
      while ( 2 )
      {
        v27 = *(_QWORD *)(a2 + 8);
        v28 = v27 + v26;
        v29 = v27 + v26 + 80;
        v30 = *(_DWORD *)(v27 + v26 + 44);
        a5 = v27 + ((v24 + v25 - 1 - (unsigned int)v24) << 6) + 80;
        do
        {
          if ( v30 != *(_DWORD *)(v29 + 28) && *(_DWORD *)(v28 + 48) == *(_DWORD *)(v29 + 32) )
          {
            v31 = *(__int64 **)v29;
            v32 = **(_QWORD **)(v28 + 16);
            if ( *(_BYTE *)(v32 + 8) == 16 )
              v32 = **(_QWORD **)(v32 + 16);
            v33 = *(_DWORD *)(v32 + 8);
            v34 = *v31;
            v35 = v33 >> 8;
            if ( *(_BYTE *)(*v31 + 8) == 16 )
              v34 = **(_QWORD **)(v34 + 16);
            if ( v35 != *(_DWORD *)(v34 + 8) >> 8 )
              return 0;
          }
          v29 += 64;
        }
        while ( v29 != a5 );
        ++v24;
        v26 += 64;
        if ( (unsigned int)v24 < v25 )
          continue;
        break;
      }
    }
  }
  if ( (v44 & (unsigned __int8)v43) != 0 )
  {
    sub_38663E0(a2, *(_QWORD *)(v7 + 400), v46 != 0, v46, (unsigned __int8)a5, v24);
    *(_BYTE *)a2 = 1;
    return (unsigned __int8)(v44 & v43);
  }
  result = v44 ^ 1u;
  LOBYTE(result) = v43 | v44 ^ 1;
  if ( (_BYTE)result )
  {
LABEL_69:
    *(_BYTE *)a2 = v44;
    return 1;
  }
  v37 = *(_QWORD **)(a2 + 8);
  *(_BYTE *)a2 = 0;
  v38 = result;
  v39 = &v37[8 * (unsigned __int64)v25];
  if ( v37 != v39 )
  {
    do
    {
      v40 = *(v39 - 6);
      v39 -= 8;
      if ( v40 != -8 && v40 != 0 && v40 != -16 )
        sub_1649B30(v39);
    }
    while ( v37 != v39 );
    result = v38;
  }
  *(_DWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 280) = 0;
  return result;
}
