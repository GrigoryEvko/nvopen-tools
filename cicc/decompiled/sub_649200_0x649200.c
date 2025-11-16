// Function: sub_649200
// Address: 0x649200
//
void __fastcall sub_649200(__int64 a1, __int64 a2, int a3, int a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 i; // rax
  __int64 v16; // rdx
  __int64 **v17; // rax
  __int64 *v18; // rsi
  int v19; // r10d
  int v20; // r15d
  int v21; // edi
  char v22; // dl
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 **j; // r12
  __int64 **v28; // r15
  __int64 *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // r8
  char v34; // al
  __m128i *v35; // rdx
  const __m128i *v36; // rcx
  int v37; // r11d
  __int64 v38; // r10
  __int64 v39; // r9
  __int64 v40; // rdi
  char v41; // si
  __m128i *v42; // rax
  char v43; // si
  __int64 v44; // r13
  __int64 v45; // rsi
  __int64 k; // rdi
  char v47; // al
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  _QWORD *v51; // rbx
  _QWORD *v52; // r12
  __int64 v53; // rdi
  __int64 v54; // [rsp+0h] [rbp-40h]
  __int64 v55; // [rsp+0h] [rbp-40h]
  __int64 v56; // [rsp+0h] [rbp-40h]
  __int64 v57; // [rsp+0h] [rbp-40h]
  __int64 v58; // [rsp+8h] [rbp-38h]
  __int64 v59; // [rsp+8h] [rbp-38h]
  __int64 v60; // [rsp+8h] [rbp-38h]
  int v61; // [rsp+8h] [rbp-38h]
  int v62; // [rsp+8h] [rbp-38h]
  __int64 v63; // [rsp+8h] [rbp-38h]
  __int64 savedregs; // [rsp+40h] [rbp+0h] BYREF

  v5 = *(_QWORD *)(a1 + 152);
  if ( v5 != a2 )
  {
    if ( dword_4F077C4 == 2 && !*(_QWORD *)(a5 + 456) )
    {
      for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v16 = *(_QWORD *)(a1 + 152);
      v17 = **(__int64 ****)(i + 168);
      if ( *(_BYTE *)(v5 + 140) == 12 )
      {
        do
          v16 = *(_QWORD *)(v16 + 160);
        while ( *(_BYTE *)(v16 + 140) == 12 );
      }
      v18 = **(__int64 ***)(v16 + 168);
      if ( v17 )
      {
        v19 = 0;
        v20 = 0;
        v21 = 0;
        do
        {
          v22 = v18[4] & 4;
          if ( ((_BYTE)v17[4] & 4) != 0 )
          {
            v21 = 1;
            if ( v22 )
              v19 = 1;
          }
          else if ( v22 )
          {
            v21 = 1;
          }
          else if ( v21 && (*((_BYTE *)v18 + 33) & 1) == 0 )
          {
            v20 = v21;
          }
          v17 = (__int64 **)*v17;
          v18 = (__int64 *)*v18;
        }
        while ( v17 );
        if ( v19 )
        {
          if ( dword_4F077BC && *(_QWORD *)(a1 + 240) )
            v49 = 3 * (unsigned int)(*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 8) + 5;
          else
            v49 = 8;
          v55 = a5;
          v61 = a4;
          sub_684AC0(v49, 307);
          a5 = v55;
          a4 = v61;
        }
        if ( v20 )
        {
          v56 = a5;
          v62 = a4;
          sub_6851C0(306, dword_4F07508);
          a5 = v56;
          a4 = v62;
        }
      }
    }
    if ( a3 | a4 )
    {
      v58 = a5;
      if ( !a3 )
      {
        v23 = v5;
        v24 = sub_8D79B0(a2, v5);
        v10 = v58;
        v11 = v24;
        if ( *(_QWORD *)(v58 + 456) )
        {
          while ( *(_BYTE *)(v5 + 140) == 12 )
            v5 = *(_QWORD *)(v5 + 160);
          for ( j = **(__int64 ****)(v5 + 168); *(_BYTE *)(v24 + 140) == 12; v24 = *(_QWORD *)(v24 + 160) )
            ;
          v28 = **(__int64 ****)(v24 + 168);
          if ( v28 && j )
          {
            do
            {
              if ( ((_BYTE)j[4] & 4) != 0 && ((_BYTE)v28[4] & 4) != 0 )
              {
                v29 = j[5];
                if ( v29 )
                {
                  if ( !v28[5] )
                  {
                    v57 = v10;
                    v63 = v11;
                    v50 = sub_73BB50(v29, v23, v25, v26, v10);
                    v10 = v57;
                    v11 = v63;
                    v28[5] = (__int64 *)v50;
                    v28[7] = j[7];
                  }
                }
              }
              j = (__int64 **)*j;
              v28 = (__int64 **)*v28;
            }
            while ( j && v28 );
          }
        }
        *(_QWORD *)(a1 + 152) = a2;
        v5 = a2;
        if ( *(_BYTE *)(v11 + 140) != 12 )
          goto LABEL_57;
        goto LABEL_53;
      }
      v9 = sub_8D79B0(v5, a2);
      v10 = v58;
      v11 = v9;
      if ( v5 != v9 )
      {
        if ( !v9 || !v5 || !dword_4F07588 || (v48 = *(_QWORD *)(v9 + 32), *(_QWORD *)(v5 + 32) != v48) || !v48 )
        {
          v12 = *(_QWORD *)(a1 + 264);
          if ( v12 == v5
            || v12 && v5 && dword_4F07588 && (v13 = *(_QWORD *)(v12 + 32), *(_QWORD *)(v5 + 32) == v13) && v13 )
          {
            v54 = v58;
            v59 = v11;
            v14 = sub_73EDA0(v5, 1);
            v11 = v59;
            v10 = v54;
            *(_QWORD *)(a1 + 264) = v14;
          }
        }
      }
      while ( *(_BYTE *)(v11 + 140) == 12 )
LABEL_53:
        v11 = *(_QWORD *)(v11 + 160);
      while ( *(_BYTE *)(v5 + 140) == 12 )
      {
        v5 = *(_QWORD *)(v5 + 160);
LABEL_57:
        ;
      }
      if ( v11 != v5 )
      {
        if ( !dword_4F07588 || (v30 = *(_QWORD *)(v11 + 32), *(_QWORD *)(v5 + 32) != v30) || !v30 )
        {
          v31 = *(_QWORD *)(v11 + 168);
          v32 = *(_QWORD *)(v5 + 168);
          v60 = v11;
          *(_QWORD *)(v10 + 296) = sub_73EDA0(v5, 0);
          *(_QWORD *)(v5 + 160) = *(_QWORD *)(v60 + 160);
          sub_5D1620(v5, v60);
          v34 = *(_BYTE *)(v31 + 16) & 2 | *(_BYTE *)(v32 + 16) & 0xFD;
          *(_BYTE *)(v32 + 16) = v34;
          *(_BYTE *)(v32 + 16) = *(_BYTE *)(v31 + 16) & 1 | v34 & 0xFE;
          if ( (*(_BYTE *)(v31 + 20) & 1) != 0 )
            *(_BYTE *)(v32 + 20) |= 1u;
          v35 = *(__m128i **)v32;
          v36 = *(const __m128i **)v31;
          if ( *(_QWORD *)v32 )
          {
            if ( v36 && v35 != v36 )
            {
              v37 = unk_4F0690C;
              if ( unk_4F0690C )
              {
                v37 = 1;
                if ( a3 )
                  v37 = *(_DWORD *)(a1 + 160) != 0;
              }
              do
              {
                v38 = v35[1].m128i_i64[0];
                v39 = v35[1].m128i_i64[1];
                v33 = v35[2].m128i_u32[1];
                v40 = v35[4].m128i_i64[1];
                v41 = (unsigned __int32)v35[2].m128i_i32[0] >> 11;
                v42 = v35;
                v35 = (__m128i *)v35->m128i_i64[0];
                *v42 = _mm_loadu_si128(v36);
                v43 = v41 & 0x7F;
                v42[1] = _mm_loadu_si128(v36 + 1);
                v42[2] = _mm_loadu_si128(v36 + 2);
                v42[3] = _mm_loadu_si128(v36 + 3);
                v42[4] = _mm_loadu_si128(v36 + 4);
                v44 = v36[5].m128i_i64[0];
                v42->m128i_i64[0] = (__int64)v35;
                v42[5].m128i_i64[0] = v44;
                v42[1].m128i_i64[0] = v38;
                if ( v37 )
                  v42[2].m128i_i32[0] = v42[2].m128i_i32[0] & 0xFFFC07FF | ((v43 & 0x7F) << 11);
                v42[1].m128i_i64[1] = v39;
                v42[2].m128i_i32[1] = v33;
                v42[4].m128i_i64[1] = v40;
                v36 = (const __m128i *)v36->m128i_i64[0];
              }
              while ( v35 );
            }
          }
          else
          {
            *(_QWORD *)v32 = v36;
          }
          if ( dword_4F077C4 == 2 )
          {
            if ( dword_4D048B8 )
              *(_QWORD *)(v32 + 56) = *(_QWORD *)(v31 + 56);
            v47 = *(_BYTE *)(v31 + 17) & 0x70 | *(_BYTE *)(v32 + 17) & 0x8F;
            *(_BYTE *)(v32 + 17) = v47;
            v35 = (__m128i *)(*(_BYTE *)(v31 + 17) & 0x80);
            *(_BYTE *)(v32 + 17) = *(_BYTE *)(v31 + 17) & 0x80 | v47 & 0x7F;
          }
          v45 = *(_QWORD *)(a1 + 264);
          if ( v45 )
          {
            for ( k = v5; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
              ;
            for ( ; *(_BYTE *)(v45 + 140) == 12; v45 = *(_QWORD *)(v45 + 160) )
              ;
            if ( v45 == k )
            {
              nullsub_2006(k, v45, v35, v36);
            }
            else
            {
              savedregs = (__int64)&savedregs;
              v51 = **(_QWORD ***)(k + 168);
              v52 = **(_QWORD ***)(v45 + 168);
              if ( v51 )
              {
                while ( v52 )
                {
                  v53 = v51[5];
                  if ( v53 && v53 == v52[5] )
                  {
                    v51[5] = sub_73BB50(v53, v45, v35, v36, v33);
                    v51 = (_QWORD *)*v51;
                    v52 = (_QWORD *)*v52;
                    if ( !v51 )
                      return;
                  }
                  else
                  {
                    v51 = (_QWORD *)*v51;
                    v52 = (_QWORD *)*v52;
                    if ( !v51 )
                      return;
                  }
                }
              }
            }
          }
        }
      }
    }
    else if ( (*(_BYTE *)(a1 + 193) & 0x10) != 0 )
    {
      *(_QWORD *)(a1 + 152) = sub_8D79B0(a2, v5);
    }
    else
    {
      *(_QWORD *)(a1 + 152) = sub_8D79B0(v5, a2);
    }
  }
}
