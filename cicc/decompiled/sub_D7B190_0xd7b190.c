// Function: sub_D7B190
// Address: 0xd7b190
//
__int64 __fastcall sub_D7B190(__int64 a1, __m128i *a2, __m128i **a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  bool v6; // zf
  __m128i **v7; // rax
  __m128i *v8; // rsi
  _QWORD *v9; // rdi
  unsigned __int64 *v10; // r13
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rsi
  unsigned __int8 v20; // cl
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 *v24; // r14
  _BYTE *v25; // rbx
  unsigned __int8 v26; // al
  _QWORD *v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // r12
  _QWORD *v30; // rax
  _QWORD *v31; // r12
  _QWORD *v32; // r13
  char v33; // dl
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int8 v38; // [rsp+27h] [rbp-209h]
  __int64 v39; // [rsp+28h] [rbp-208h]
  __int64 v40; // [rsp+30h] [rbp-200h]
  unsigned __int64 v41; // [rsp+38h] [rbp-1F8h]
  _QWORD *v42; // [rsp+40h] [rbp-1F0h]
  __int64 v44; // [rsp+50h] [rbp-1E0h]
  __int64 v45; // [rsp+50h] [rbp-1E0h]
  unsigned __int64 v46; // [rsp+58h] [rbp-1D8h]
  __int64 *v47; // [rsp+58h] [rbp-1D8h]
  __int64 v48; // [rsp+60h] [rbp-1D0h] BYREF
  unsigned __int64 v49; // [rsp+68h] [rbp-1C8h] BYREF
  __m128i v50; // [rsp+70h] [rbp-1C0h] BYREF
  _QWORD *v51; // [rsp+80h] [rbp-1B0h]
  _QWORD *v52; // [rsp+88h] [rbp-1A8h]
  __int64 v53; // [rsp+90h] [rbp-1A0h]
  __m128i v54; // [rsp+A0h] [rbp-190h] BYREF
  _QWORD *v55; // [rsp+B0h] [rbp-180h] BYREF
  _QWORD *v56; // [rsp+B8h] [rbp-178h]
  __int64 v57; // [rsp+C0h] [rbp-170h]
  _BYTE *v58; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v59; // [rsp+F8h] [rbp-138h]
  _BYTE v60[304]; // [rsp+100h] [rbp-130h] BYREF

  v6 = *(_BYTE *)(a4 + 28) == 0;
  v58 = v60;
  v40 = (__int64)a3;
  v39 = a4;
  v59 = 0x2000000000LL;
  if ( v6 )
    goto LABEL_27;
  v7 = *(__m128i ***)(a4 + 8);
  a4 = *(unsigned int *)(a4 + 20);
  a3 = &v7[a4];
  if ( v7 == a3 )
  {
LABEL_26:
    if ( (unsigned int)a4 < *(_DWORD *)(v39 + 16) )
    {
      *(_DWORD *)(v39 + 20) = a4 + 1;
      *a3 = a2;
      v17 = (unsigned int)v59;
      ++*(_QWORD *)v39;
      goto LABEL_81;
    }
LABEL_27:
    sub_C8CC70(v39, (__int64)a2, (__int64)a3, a4, (__int64)a5, a6);
    v16 = v15;
    v17 = (unsigned int)v59;
    if ( !(_BYTE)v16 )
    {
LABEL_28:
      v38 = 0;
      while ( 1 )
      {
        do
        {
          if ( !(_DWORD)v17 )
            goto LABEL_7;
          v18 = (unsigned int)v17;
          LODWORD(v17) = v17 - 1;
          v45 = 0;
          v19 = *(_QWORD *)&v58[8 * v18 - 8];
          LODWORD(v59) = v17;
          v47 = (__int64 *)v19;
          if ( *(_BYTE *)v19 > 0x1Cu )
          {
            v20 = *(_BYTE *)v19 - 34;
            if ( v20 <= 0x33u )
            {
              v21 = 0;
              if ( ((0x8000000000041uLL >> v20) & 1) != 0 )
                v21 = v19;
              v45 = v21;
            }
          }
          v22 = 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF);
          v23 = v19 - v22;
          if ( (*(_BYTE *)(v19 + 7) & 0x40) != 0 )
          {
            v23 = *(_QWORD *)(v19 - 8);
            v47 = (__int64 *)(v23 + v22);
          }
        }
        while ( v47 == (__int64 *)v23 );
        v24 = (__int64 *)v23;
        do
        {
          while ( 1 )
          {
            while ( 1 )
            {
              while ( 1 )
              {
                v25 = (_BYTE *)*v24;
                v26 = *(_BYTE *)*v24;
                if ( v26 <= 0x1Cu )
                  break;
LABEL_40:
                v27 = (_QWORD *)v39;
                if ( *(_BYTE *)(v39 + 28) )
                {
                  v28 = *(_QWORD **)(v39 + 8);
                  v23 = *(unsigned int *)(v39 + 20);
                  v27 = &v28[v23];
                  if ( v28 != v27 )
                  {
                    while ( v25 != (_BYTE *)*v28 )
                    {
                      if ( v27 == ++v28 )
                        goto LABEL_77;
                    }
                    goto LABEL_45;
                  }
LABEL_77:
                  if ( (unsigned int)v23 < *(_DWORD *)(v39 + 16) )
                  {
                    *(_DWORD *)(v39 + 20) = v23 + 1;
                    *v27 = v25;
                    ++*(_QWORD *)v39;
                    goto LABEL_73;
                  }
                }
                sub_C8CC70(v39, *v24, (__int64)v27, v23, v16, a6);
                if ( v33 )
                {
LABEL_73:
                  v34 = (unsigned int)v59;
                  v23 = HIDWORD(v59);
                  v35 = (unsigned int)v59 + 1LL;
                  if ( v35 > HIDWORD(v59) )
                  {
                    sub_C8D5F0((__int64)&v58, v60, v35, 8u, v16, a6);
                    v34 = (unsigned int)v59;
                  }
                  v24 += 4;
                  *(_QWORD *)&v58[8 * v34] = v25;
                  LODWORD(v59) = v59 + 1;
                  if ( v47 == v24 )
                    goto LABEL_46;
                }
                else
                {
LABEL_45:
                  v24 += 4;
                  if ( v47 == v24 )
                    goto LABEL_46;
                }
              }
              if ( v26 > 0x15u )
                goto LABEL_45;
              if ( v26 != 4 )
                break;
              v38 = 1;
              v24 += 4;
              if ( v47 == v24 )
                goto LABEL_46;
            }
            if ( v26 > 3u )
              goto LABEL_40;
            if ( v45 )
            {
              v23 = v45;
              if ( v24 == (__int64 *)(v45 - 32) )
                goto LABEL_45;
            }
            if ( v26 != 2 || (v25[32] & 0xFu) - 7 > 1 )
              break;
            v24 += 4;
            *a5 = 1;
            if ( v47 == v24 )
              goto LABEL_46;
          }
          sub_B2F930(&v54, *v24);
          v29 = sub_B2F650(v54.m128i_i64[0], v54.m128i_i64[1]);
          if ( (_QWORD **)v54.m128i_i64[0] != &v55 )
            j_j___libc_free_0(v54.m128i_i64[0], (char *)v55 + 1);
          v50.m128i_i64[0] = v29;
          if ( *(_BYTE *)(a1 + 343) )
          {
            v54.m128i_i64[0] = 0;
          }
          else
          {
            v54.m128i_i64[1] = 0;
            v54.m128i_i64[0] = (__int64)byte_3F871B3;
          }
          v55 = 0;
          v56 = 0;
          v57 = 0;
          v30 = sub_9CA390((_QWORD *)a1, (unsigned __int64 *)&v50, &v54);
          v31 = v56;
          v32 = v55;
          v42 = v30;
          v41 = (unsigned __int64)(v30 + 4);
          if ( v56 != v55 )
          {
            do
            {
              if ( *v32 )
                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v32 + 8LL))(*v32);
              ++v32;
            }
            while ( v31 != v32 );
            v32 = v55;
          }
          if ( v32 )
            j_j___libc_free_0(v32, v57 - (_QWORD)v32);
          v24 += 4;
          v42[5] = v25;
          v54.m128i_i64[0] = *(unsigned __int8 *)(a1 + 343) | v41 & 0xFFFFFFFFFFFFFFF8LL;
          sub_D7AF10(v40, &v54);
        }
        while ( v47 != v24 );
LABEL_46:
        LODWORD(v17) = v59;
      }
    }
LABEL_81:
    v16 = v17 + 1;
    if ( v17 + 1 > (unsigned __int64)HIDWORD(v59) )
    {
      sub_C8D5F0((__int64)&v58, v60, v17 + 1, 8u, v16, a6);
      v17 = (unsigned int)v59;
    }
    *(_QWORD *)&v58[8 * v17] = a2;
    LODWORD(v17) = v59 + 1;
    LODWORD(v59) = v59 + 1;
    goto LABEL_28;
  }
  while ( a2 != *v7 )
  {
    if ( a3 == ++v7 )
      goto LABEL_26;
  }
  v38 = 0;
LABEL_7:
  v8 = a2;
  if ( a2->m128i_i8[0] > 0x1Cu )
  {
    v48 = 0;
    sub_ED2710(&v54, a2, 2, LODWORD(qword_4F8EA48[8]), &v48, 0);
    v9 = (_QWORD *)v54.m128i_i64[0];
    v44 = v54.m128i_i64[0] + 16LL * v54.m128i_u32[2];
    if ( v44 != v54.m128i_i64[0] )
    {
      v10 = (unsigned __int64 *)v54.m128i_i64[0];
      do
      {
        v6 = *(_BYTE *)(a1 + 343) == 0;
        v49 = *v10;
        if ( v6 )
        {
          v50.m128i_i64[1] = 0;
          v50.m128i_i64[0] = (__int64)byte_3F871B3;
        }
        else
        {
          v50.m128i_i64[0] = 0;
        }
        v51 = 0;
        v52 = 0;
        v53 = 0;
        v11 = sub_9CA390((_QWORD *)a1, &v49, &v50);
        v12 = v52;
        v13 = v51;
        v46 = (unsigned __int64)(v11 + 4);
        if ( v52 != v51 )
        {
          do
          {
            if ( *v13 )
              (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v13 + 8LL))(*v13);
            ++v13;
          }
          while ( v12 != v13 );
          v13 = v51;
        }
        if ( v13 )
          j_j___libc_free_0(v13, v53 - (_QWORD)v13);
        v8 = &v50;
        v10 += 2;
        v50.m128i_i64[0] = *(unsigned __int8 *)(a1 + 343) | v46 & 0xFFFFFFFFFFFFFFF8LL;
        sub_D7AF10(v40, &v50);
      }
      while ( (unsigned __int64 *)v44 != v10 );
      v9 = (_QWORD *)v54.m128i_i64[0];
    }
    if ( v9 != &v55 )
      _libc_free(v9, v8);
  }
  if ( v58 != v60 )
    _libc_free(v58, v8);
  return v38;
}
