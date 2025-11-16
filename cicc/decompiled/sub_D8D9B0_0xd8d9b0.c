// Function: sub_D8D9B0
// Address: 0xd8d9b0
//
__int64 *__fastcall sub_D8D9B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  const void **v4; // r12
  __int64 v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rbx
  _QWORD *v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  unsigned __int64 v17; // rax
  __int64 v18; // r12
  unsigned int v19; // eax
  unsigned int v20; // eax
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rbx
  __int64 v27; // r13
  __int64 v28; // r14
  __int64 v29; // r15
  unsigned __int64 v30; // rax
  __int64 *v31; // r12
  __int64 *v32; // rdi
  __int64 v34; // r12
  __int64 v35; // r14
  __int64 v36; // rbx
  __int64 v37; // r13
  unsigned int v38; // eax
  unsigned int v39; // eax
  unsigned int v40; // eax
  const void **v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // r15
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 v47; // [rsp+10h] [rbp-B0h]
  const void **v48; // [rsp+20h] [rbp-A0h]
  __int64 v49; // [rsp+28h] [rbp-98h]
  unsigned __int64 v50; // [rsp+30h] [rbp-90h]
  _QWORD *v51; // [rsp+38h] [rbp-88h]
  __int64 v52; // [rsp+38h] [rbp-88h]
  __int64 v53; // [rsp+40h] [rbp-80h]
  __int64 v55; // [rsp+58h] [rbp-68h] BYREF
  __m128i v56; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v57; // [rsp+70h] [rbp-50h] BYREF
  _QWORD *v58; // [rsp+78h] [rbp-48h]
  __int64 v59; // [rsp+80h] [rbp-40h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v3 = sub_D8C9C0(a2, a2, a3);
  v47 = v3 + 56;
  v49 = *(_QWORD *)(v3 + 72);
  if ( v49 != v3 + 56 )
  {
    while ( 1 )
    {
      v4 = (const void **)(v49 + 40);
      if ( !sub_AAF760(v49 + 40) )
        break;
LABEL_3:
      v49 = sub_220EF30(v49);
      if ( v47 == v49 )
        goto LABEL_54;
    }
    v5 = a1[1];
    if ( v5 == a1[2] )
    {
      sub_D8A3E0(a1, a1[1], (unsigned int *)(v49 + 32), (__int64)v4);
      v53 = a1[1];
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = *(unsigned int *)(v49 + 32);
        v6 = *(_DWORD *)(v49 + 48);
        *(_DWORD *)(v5 + 16) = v6;
        if ( v6 > 0x40 )
          sub_C43780(v5 + 8, v4);
        else
          *(_QWORD *)(v5 + 8) = *(_QWORD *)(v49 + 40);
        v7 = *(_DWORD *)(v49 + 64);
        *(_DWORD *)(v5 + 32) = v7;
        if ( v7 > 0x40 )
          sub_C43780(v5 + 24, (const void **)(v49 + 56));
        else
          *(_QWORD *)(v5 + 24) = *(_QWORD *)(v49 + 56);
        *(_QWORD *)(v5 + 40) = 0;
        *(_QWORD *)(v5 + 48) = 0;
        *(_QWORD *)(v5 + 56) = 0;
        v5 = a1[1];
      }
      v53 = v5 + 64;
      a1[1] = v5 + 64;
    }
    v8 = *(_QWORD *)(v49 + 160);
    if ( v8 > 0x2AAAAAAAAAAAAAALL )
      sub_4262D8((__int64)"vector::reserve");
    v9 = *(_QWORD *)(v53 - 24);
    v10 = *(_QWORD *)(v53 - 8) - v9;
    if ( v8 <= 0xAAAAAAAAAAAAAAABLL * (v10 >> 4) )
    {
LABEL_15:
      v11 = *(_QWORD *)(v49 + 144);
      if ( v49 + 128 != v11 )
      {
        while ( 1 )
        {
          v48 = (const void **)(v11 + 48);
          if ( sub_AAF760(v11 + 48) )
            break;
          v12 = *(_QWORD *)(v11 + 32);
          sub_B2F930(&v56, v12);
          v13 = sub_B2F650(v56.m128i_i64[0], v56.m128i_i64[1]);
          if ( (_QWORD **)v56.m128i_i64[0] != &v57 )
            j_j___libc_free_0(v56.m128i_i64[0], (char *)v57 + 1);
          v55 = v13;
          if ( *(_BYTE *)(a3 + 343) )
          {
            v56.m128i_i64[0] = 0;
          }
          else
          {
            v56.m128i_i64[1] = 0;
            v56.m128i_i64[0] = (__int64)byte_3F871B3;
          }
          v57 = 0;
          v58 = 0;
          v59 = 0;
          v14 = sub_9CA390((_QWORD *)a3, (unsigned __int64 *)&v55, &v56);
          v15 = v58;
          v16 = v57;
          v51 = v14;
          v50 = (unsigned __int64)(v14 + 4);
          if ( v58 != v57 )
          {
            do
            {
              if ( *v16 )
                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v16 + 8LL))(*v16);
              ++v16;
            }
            while ( v15 != v16 );
            v16 = v57;
          }
          if ( v16 )
            j_j___libc_free_0(v16, v59 - (_QWORD)v16);
          v51[5] = v12;
          v17 = *(unsigned __int8 *)(a3 + 343) | v50 & 0xFFFFFFFFFFFFFFF8LL;
          v56.m128i_i64[0] = v17;
          v18 = *(_QWORD *)(v53 - 16);
          if ( v18 == *(_QWORD *)(v53 - 8) )
          {
            sub_D8A980((__int64 *)(v53 - 24), *(_QWORD *)(v53 - 16), (__int64 *)(v11 + 40), &v56, (__int64)v48);
          }
          else
          {
            if ( v18 )
            {
              *(_QWORD *)v18 = *(_QWORD *)(v11 + 40);
              *(_QWORD *)(v18 + 8) = v17;
              v19 = *(_DWORD *)(v11 + 56);
              *(_DWORD *)(v18 + 24) = v19;
              if ( v19 > 0x40 )
                sub_C43780(v18 + 16, v48);
              else
                *(_QWORD *)(v18 + 16) = *(_QWORD *)(v11 + 48);
              v20 = *(_DWORD *)(v11 + 72);
              *(_DWORD *)(v18 + 40) = v20;
              if ( v20 > 0x40 )
                sub_C43780(v18 + 32, (const void **)(v11 + 64));
              else
                *(_QWORD *)(v18 + 32) = *(_QWORD *)(v11 + 64);
              v18 = *(_QWORD *)(v53 - 16);
            }
            *(_QWORD *)(v53 - 16) = v18 + 48;
          }
          v11 = sub_220EF30(v11);
          if ( v49 + 128 == v11 )
            goto LABEL_3;
        }
        v21 = a1[1];
        a1[1] = v21 - 64;
        v22 = *(_QWORD *)(v21 - 16);
        v23 = *(_QWORD *)(v21 - 24);
        if ( v22 != v23 )
        {
          do
          {
            if ( *(_DWORD *)(v23 + 40) > 0x40u )
            {
              v24 = *(_QWORD *)(v23 + 32);
              if ( v24 )
                j_j___libc_free_0_0(v24);
            }
            if ( *(_DWORD *)(v23 + 24) > 0x40u )
            {
              v25 = *(_QWORD *)(v23 + 16);
              if ( v25 )
                j_j___libc_free_0_0(v25);
            }
            v23 += 48;
          }
          while ( v22 != v23 );
          v23 = *(_QWORD *)(v21 - 24);
        }
        if ( v23 )
          j_j___libc_free_0(v23, *(_QWORD *)(v21 - 8) - v23);
        sub_969240((__int64 *)(v21 - 40));
        sub_969240((__int64 *)(v21 - 56));
      }
      goto LABEL_3;
    }
    v34 = 48 * v8;
    v35 = *(_QWORD *)(v53 - 16);
    v52 = v35 - v9;
    if ( v8 )
    {
      v36 = sub_22077B0(48 * v8);
      if ( v9 != v35 )
        goto LABEL_65;
LABEL_73:
      v43 = *(_QWORD *)(v53 - 16);
      v35 = *(_QWORD *)(v53 - 24);
      if ( v43 == v35 )
      {
        v10 = *(_QWORD *)(v53 - 8) - v35;
      }
      else
      {
        do
        {
          if ( *(_DWORD *)(v35 + 40) > 0x40u )
          {
            v44 = *(_QWORD *)(v35 + 32);
            if ( v44 )
              j_j___libc_free_0_0(v44);
          }
          if ( *(_DWORD *)(v35 + 24) > 0x40u )
          {
            v45 = *(_QWORD *)(v35 + 16);
            if ( v45 )
              j_j___libc_free_0_0(v45);
          }
          v35 += 48;
        }
        while ( v43 != v35 );
        v35 = *(_QWORD *)(v53 - 24);
        v10 = *(_QWORD *)(v53 - 8) - v35;
      }
      goto LABEL_82;
    }
    v36 = 0;
    if ( v9 == v35 )
    {
LABEL_82:
      if ( v35 )
        j_j___libc_free_0(v35, v10);
      *(_QWORD *)(v53 - 24) = v36;
      *(_QWORD *)(v53 - 16) = v36 + v52;
      *(_QWORD *)(v53 - 8) = v34 + v36;
      goto LABEL_15;
    }
LABEL_65:
    v37 = v36;
    while ( 1 )
    {
      if ( !v37 )
        goto LABEL_68;
      *(_QWORD *)v37 = *(_QWORD *)v9;
      *(_QWORD *)(v37 + 8) = *(_QWORD *)(v9 + 8);
      v39 = *(_DWORD *)(v9 + 24);
      *(_DWORD *)(v37 + 24) = v39;
      if ( v39 > 0x40 )
        break;
      *(_QWORD *)(v37 + 16) = *(_QWORD *)(v9 + 16);
      v38 = *(_DWORD *)(v9 + 40);
      *(_DWORD *)(v37 + 40) = v38;
      if ( v38 > 0x40 )
      {
LABEL_72:
        v41 = (const void **)(v9 + 32);
        v42 = v37 + 32;
        v9 += 48;
        v37 += 48;
        sub_C43780(v42, v41);
        if ( v35 == v9 )
          goto LABEL_73;
      }
      else
      {
LABEL_67:
        *(_QWORD *)(v37 + 32) = *(_QWORD *)(v9 + 32);
LABEL_68:
        v9 += 48;
        v37 += 48;
        if ( v35 == v9 )
          goto LABEL_73;
      }
    }
    sub_C43780(v37 + 16, (const void **)(v9 + 16));
    v40 = *(_DWORD *)(v9 + 40);
    *(_DWORD *)(v37 + 40) = v40;
    if ( v40 > 0x40 )
      goto LABEL_72;
    goto LABEL_67;
  }
LABEL_54:
  v26 = *a1;
  v27 = a1[1];
  if ( v27 != *a1 )
  {
    do
    {
      while ( 1 )
      {
        v28 = *(_QWORD *)(v26 + 48);
        v29 = *(_QWORD *)(v26 + 40);
        if ( v28 != v29 )
          break;
LABEL_59:
        v26 += 64;
        if ( v27 == v26 )
          return a1;
      }
      _BitScanReverse64(&v30, 0xAAAAAAAAAAAAAAABLL * ((v28 - v29) >> 4));
      sub_D8D390(*(_QWORD *)(v26 + 40), *(__int64 **)(v26 + 48), 2LL * (int)(63 - (v30 ^ 0x3F)));
      if ( v28 - v29 > 768 )
      {
        v31 = (__int64 *)(v29 + 768);
        sub_D87030(v29, v29 + 768);
        if ( v28 != v29 + 768 )
        {
          do
          {
            v32 = v31;
            v31 += 6;
            sub_D86EE0(v32);
          }
          while ( (__int64 *)v28 != v31 );
        }
        goto LABEL_59;
      }
      v26 += 64;
      sub_D87030(v29, v28);
    }
    while ( v27 != v26 );
  }
  return a1;
}
