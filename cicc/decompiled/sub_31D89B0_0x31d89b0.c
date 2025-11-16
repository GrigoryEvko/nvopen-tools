// Function: sub_31D89B0
// Address: 0x31d89b0
//
__int16 __fastcall sub_31D89B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r15
  __int64 v7; // r13
  __int64 (*v8)(void); // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // r8
  void *v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 (*v16)(void); // rax
  __int64 v17; // rcx
  _WORD *v18; // rsi
  int v19; // ebx
  void *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // r15
  __m128i *v24; // rdx
  __m128i v25; // xmm0
  size_t v26; // rcx
  void *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdi
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int64 v32; // r8
  void *v33; // rdx
  __m128i *v34; // rdx
  __m128i si128; // xmm0
  __int64 v36; // rdx
  unsigned __int64 v37; // r8
  __m128i *v38; // rdx
  __m128i v39; // xmm0
  int v40; // esi
  int v41; // edx
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rdx
  __m128i *v45; // rdx
  __m128i v46; // xmm0
  __int64 v47; // rdx
  signed __int64 v49; // [rsp+8h] [rbp-B8h]
  signed __int64 v50; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v51; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __m128i v53[5]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v54; // [rsp+78h] [rbp-48h]
  __int64 v55; // [rsp+88h] [rbp-38h]

  v3 = 0;
  v7 = sub_2E88D60(a1);
  v8 = *(__int64 (**)(void))(**(_QWORD **)(v7 + 16) + 128LL);
  if ( v8 != sub_2DAC790 )
    v3 = (__int64 *)v8();
  v9 = sub_2E8E4C0(a1, (__int64)v3);
  v52 = v10;
  v51 = v9;
  if ( (_BYTE)v10 )
  {
    v11 = v51 & 0x3FFFFFFFFFFFFFFFLL;
    if ( _bittest64((const signed __int64 *)&v51, 0x3Eu) )
    {
      v28 = *(_QWORD *)(a3 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v28) <= 8 )
      {
        v49 = v51 & 0x3FFFFFFFFFFFFFFFLL;
        sub_CB6200(a3, "vscale x ", 9u);
        v11 = v49;
      }
      else
      {
        *(_BYTE *)(v28 + 8) = 32;
        *(_QWORD *)v28 = 0x7820656C61637376LL;
        *(_QWORD *)(a3 + 32) += 9LL;
      }
    }
    sub_CB59D0(a3, v11);
    v12 = *(void **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v12 <= 0xCu )
    {
      LOWORD(v13) = sub_CB6200(a3, "-byte Reload\n", 0xDu);
    }
    else
    {
      LOWORD(v13) = 25133;
      qmemcpy(v12, "-byte Reload\n", 13);
      *(_QWORD *)(a3 + 32) += 13LL;
    }
  }
  else
  {
    v21 = sub_2E8E600(a1, v3);
    v52 = v22;
    v51 = v21;
    if ( (_BYTE)v22 )
    {
      LOWORD(v13) = v51;
      if ( v51 == -1 || v51 == 0xBFFFFFFFFFFFFFFELL )
      {
        v34 = *(__m128i **)(a3 + 32);
        if ( *(_QWORD *)(a3 + 24) - (_QWORD)v34 <= 0x1Au )
        {
          LOWORD(v13) = sub_CB6200(a3, "Unknown-size Folded Reload\n", 0x1Bu);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_44D40F0);
          LOWORD(v13) = 25956;
          qmemcpy(&v34[1], "ded Reload\n", 11);
          *v34 = si128;
          *(_QWORD *)(a3 + 32) += 27LL;
        }
      }
      else
      {
        v23 = v51 & 0x3FFFFFFFFFFFFFFFLL;
        if ( (v51 & 0x3FFFFFFFFFFFFFFFLL) != 0 )
        {
          if ( (v51 & 0x4000000000000000LL) != 0 )
          {
            v43 = *(_QWORD *)(a3 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v43) <= 8 )
            {
              sub_CB6200(a3, "vscale x ", 9u);
            }
            else
            {
              *(_BYTE *)(v43 + 8) = 32;
              *(_QWORD *)v43 = 0x7820656C61637376LL;
              *(_QWORD *)(a3 + 32) += 9LL;
            }
          }
          sub_CB59D0(a3, v23);
          v24 = *(__m128i **)(a3 + 32);
          v13 = *(_QWORD *)(a3 + 24) - (_QWORD)v24;
          if ( v13 <= 0x13 )
          {
            LOWORD(v13) = sub_CB6200(a3, "-byte Folded Reload\n", 0x14u);
          }
          else
          {
            v25 = _mm_load_si128((const __m128i *)&xmmword_44D4100);
            v24[1].m128i_i32[0] = 174350703;
            *v24 = v25;
            *(_QWORD *)(a3 + 32) += 20LL;
          }
        }
      }
    }
    else
    {
      v30 = sub_2E8E2F0(a1, (__int64)v3);
      v52 = v31;
      v51 = v30;
      if ( (_BYTE)v31 )
      {
        v32 = v51 & 0x3FFFFFFFFFFFFFFFLL;
        if ( _bittest64((const signed __int64 *)&v51, 0x3Eu) )
        {
          v44 = *(_QWORD *)(a3 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v44) <= 8 )
          {
            v50 = v51 & 0x3FFFFFFFFFFFFFFFLL;
            sub_CB6200(a3, "vscale x ", 9u);
            v32 = v50;
          }
          else
          {
            *(_BYTE *)(v44 + 8) = 32;
            *(_QWORD *)v44 = 0x7820656C61637376LL;
            *(_QWORD *)(a3 + 32) += 9LL;
          }
        }
        sub_CB59D0(a3, v32);
        v33 = *(void **)(a3 + 32);
        if ( *(_QWORD *)(a3 + 24) - (_QWORD)v33 <= 0xBu )
        {
          LOWORD(v13) = sub_CB6200(a3, "-byte Spill\n", 0xCu);
        }
        else
        {
          LOWORD(v13) = 25133;
          qmemcpy(v33, "-byte Spill\n", 12);
          *(_QWORD *)(a3 + 32) += 12LL;
        }
      }
      else
      {
        v13 = sub_2E8E430(a1, v3);
        v52 = v36;
        v51 = v13;
        if ( (_BYTE)v36 )
        {
          LOWORD(v13) = v51;
          if ( v51 == 0xBFFFFFFFFFFFFFFELL || v51 == -1 )
          {
            v45 = *(__m128i **)(a3 + 32);
            if ( *(_QWORD *)(a3 + 24) - (_QWORD)v45 <= 0x19u )
            {
              LOWORD(v13) = sub_CB6200(a3, "Unknown-size Folded Spill\n", 0x1Au);
            }
            else
            {
              v46 = _mm_load_si128((const __m128i *)&xmmword_44D40F0);
              LOWORD(v13) = 25956;
              qmemcpy(&v45[1], "ded Spill\n", 10);
              *v45 = v46;
              *(_QWORD *)(a3 + 32) += 26LL;
            }
          }
          else
          {
            v37 = v51 & 0x3FFFFFFFFFFFFFFFLL;
            if ( (v51 & 0x3FFFFFFFFFFFFFFFLL) != 0 )
            {
              if ( (v51 & 0x4000000000000000LL) != 0 )
              {
                v47 = *(_QWORD *)(a3 + 32);
                if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v47) <= 8 )
                {
                  sub_CB6200(a3, "vscale x ", 9u);
                  v37 = v51 & 0x3FFFFFFFFFFFFFFFLL;
                }
                else
                {
                  *(_BYTE *)(v47 + 8) = 32;
                  *(_QWORD *)v47 = 0x7820656C61637376LL;
                  *(_QWORD *)(a3 + 32) += 9LL;
                }
              }
              sub_CB59D0(a3, v37);
              v38 = *(__m128i **)(a3 + 32);
              v13 = *(_QWORD *)(a3 + 24) - (_QWORD)v38;
              if ( v13 <= 0x12 )
              {
                LOWORD(v13) = sub_CB6200(a3, "-byte Folded Spill\n", 0x13u);
              }
              else
              {
                v39 = _mm_load_si128((const __m128i *)&xmmword_44D4110);
                v38[1].m128i_i8[2] = 10;
                v38[1].m128i_i16[0] = 27756;
                *v38 = v39;
                *(_QWORD *)(a3 + 32) += 19LL;
              }
            }
          }
        }
      }
    }
  }
  if ( (*(_BYTE *)(a1 + 47) & 1) != 0 )
  {
    v27 = *(void **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v27 <= 0xDu )
    {
      LOWORD(v13) = sub_CB6200(a3, " Reload Reuse\n", 0xEu);
    }
    else
    {
      LOWORD(v13) = 21024;
      qmemcpy(v27, " Reload Reuse\n", 14);
      *(_QWORD *)(a3 + 32) += 14LL;
    }
  }
  if ( (_BYTE)qword_5035E08 )
  {
    v14 = *(_QWORD *)(v7 + 16);
    v15 = 0;
    v16 = *(__int64 (**)(void))(*(_QWORD *)v14 + 128LL);
    if ( v16 != sub_2DAC790 )
      v15 = v16();
    v13 = *(_QWORD *)(a2 + 200);
    v17 = *(_QWORD *)(v13 + 40);
    if ( v17 )
    {
      v18 = (_WORD *)(v17
                    + 14LL * *(unsigned __int16 *)(*(_QWORD *)(v15 + 8) - 40LL * *(unsigned __int16 *)(a1 + 68) + 6));
      LOWORD(v13) = *v18 & 0x1FFF;
      if ( (_WORD)v13 == 0x1FFF )
        return v13;
      LODWORD(v13) = sub_106FD90(a2, (__int64)v18);
      v19 = v13;
    }
    else
    {
      v26 = *(_QWORD *)(a2 + 72);
      if ( !v26 )
        return v13;
      sub_EA1150(v53, (_QWORD *)a2, *(_DWORD **)(a2 + 64), v26);
      v13 = *(unsigned __int16 *)(*(_QWORD *)(v15 + 8) - 40LL * *(unsigned __int16 *)(a1 + 68) + 6);
      v40 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
      if ( !v40 )
        return v13;
      v13 *= 5LL;
      v19 = 0;
      v41 = 0;
      v42 = v55 + 2 * v13;
      do
      {
        if ( v55 )
        {
          v13 = v41 + (unsigned int)*(unsigned __int16 *)(v42 + 6);
          if ( *(unsigned __int16 *)(v42 + 8) > (unsigned int)v13 )
          {
            LODWORD(v13) = *(_DWORD *)(v54 + 4 * v13);
            if ( v19 < (unsigned int)v13 )
              v19 = v13;
          }
        }
        ++v41;
      }
      while ( v40 != v41 );
    }
    if ( v19 > 1 )
    {
      v20 = *(void **)(a3 + 32);
      if ( *(_QWORD *)(a3 + 24) - (_QWORD)v20 <= 9u )
      {
        a3 = sub_CB6200(a3, " Latency: ", 0xAu);
      }
      else
      {
        qmemcpy(v20, " Latency: ", 10);
        *(_QWORD *)(a3 + 32) += 10LL;
      }
      v29 = sub_CB59F0(a3, v19);
      v13 = *(_QWORD *)(v29 + 32);
      if ( *(_QWORD *)(v29 + 24) == v13 )
      {
        LOWORD(v13) = sub_CB6200(v29, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *(_BYTE *)v13 = 10;
        ++*(_QWORD *)(v29 + 32);
      }
    }
  }
  return v13;
}
