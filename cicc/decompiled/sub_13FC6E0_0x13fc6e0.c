// Function: sub_13FC6E0
// Address: 0x13fc6e0
//
void __fastcall sub_13FC6E0(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  __int64 v6; // rdx
  const char *v7; // rsi
  __int64 v8; // r14
  void *v9; // rdx
  _QWORD *v10; // rdx
  _QWORD *v11; // r14
  _QWORD *v12; // rbx
  __m128i *v13; // rdx
  __m128i v14; // xmm0
  _BYTE *v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rdx
  _WORD *v20; // rdx
  __int64 v21; // rax
  _WORD *v22; // rdx
  void *v23; // rdx
  _BYTE *v24; // r13
  _QWORD *v25; // rbx
  __m128i *v26; // rdx
  __m128i si128; // xmm0
  _BYTE *v28; // [rsp+0h] [rbp-80h] BYREF
  __int64 v29; // [rsp+8h] [rbp-78h]
  _BYTE v30[112]; // [rsp+10h] [rbp-70h] BYREF

  if ( (unsigned __int8)sub_160E720() )
  {
    v18 = sub_16E7EE0(a2, *(const char **)a3, *(_QWORD *)(a3 + 8));
    v19 = *(_QWORD **)(v18 + 24);
    if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 7u )
    {
      sub_16E7EE0(v18, " (loop: ", 8);
    }
    else
    {
      *v19 = 0x203A706F6F6C2820LL;
      *(_QWORD *)(v18 + 24) += 8LL;
    }
    sub_15537D0(**(_QWORD **)(a1 + 32), a2, 0);
    v20 = *(_WORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v20 <= 1u )
    {
      sub_16E7EE0(a2, ")\n", 2);
    }
    else
    {
      *v20 = 2601;
      *(_QWORD *)(a2 + 24) += 2LL;
    }
    v21 = sub_157EB90(**(_QWORD **)(a1 + 32));
    sub_155BB10(v21, a2, 0, 0, 0);
  }
  else
  {
    v5 = sub_160E730();
    v6 = *(_QWORD *)(a3 + 8);
    v7 = *(const char **)a3;
    if ( !v5 )
    {
      sub_16E7EE0(a2, v7, v6);
      v8 = sub_13FC520(a1);
      if ( v8 )
      {
        v9 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v9 <= 0xCu )
        {
          sub_16E7EE0(a2, "\n; Preheader:", 13);
        }
        else
        {
          qmemcpy(v9, "\n; Preheader:", 13);
          *(_QWORD *)(a2 + 24) += 13LL;
        }
        sub_155C2B0(v8, a2, 0);
        v10 = *(_QWORD **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v10 <= 7u )
        {
          sub_16E7EE0(a2, "\n; Loop:", 8);
        }
        else
        {
          *v10 = 0x3A706F6F4C203B0ALL;
          *(_QWORD *)(a2 + 24) += 8LL;
        }
      }
      v11 = *(_QWORD **)(a1 + 40);
      v12 = *(_QWORD **)(a1 + 32);
      if ( v12 == v11 )
      {
LABEL_15:
        v28 = v30;
        v29 = 0x800000000LL;
        sub_13F9EC0(a1, (__int64)&v28);
        if ( (_DWORD)v29 )
        {
          v23 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v23 <= 0xDu )
          {
            sub_16E7EE0(a2, "\n; Exit blocks", 14);
          }
          else
          {
            qmemcpy(v23, "\n; Exit blocks", 14);
            *(_QWORD *)(a2 + 24) += 14LL;
          }
          v15 = v28;
          v24 = &v28[8 * (unsigned int)v29];
          if ( v24 == v28 )
            goto LABEL_17;
          v25 = v28;
          do
          {
            if ( *v25 )
            {
              sub_155C2B0(*v25, a2, 0);
            }
            else
            {
              v26 = *(__m128i **)(a2 + 24);
              if ( *(_QWORD *)(a2 + 16) - (_QWORD)v26 <= 0x14u )
              {
                sub_16E7EE0(a2, "Printing <null> block", 21);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_4289F10);
                v26[1].m128i_i32[0] = 1668246626;
                v26[1].m128i_i8[4] = 107;
                *v26 = si128;
                *(_QWORD *)(a2 + 24) += 21LL;
              }
            }
            ++v25;
          }
          while ( v24 != (_BYTE *)v25 );
        }
        v15 = v28;
LABEL_17:
        if ( v15 != v30 )
          _libc_free((unsigned __int64)v15);
        return;
      }
      while ( 1 )
      {
        while ( *v12 )
        {
          sub_155C2B0(*v12, a2, 0);
LABEL_11:
          if ( v11 == ++v12 )
            goto LABEL_15;
        }
        v13 = *(__m128i **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v13 <= 0x14u )
        {
          sub_16E7EE0(a2, "Printing <null> block", 21);
          goto LABEL_11;
        }
        v14 = _mm_load_si128((const __m128i *)&xmmword_4289F10);
        ++v12;
        v13[1].m128i_i32[0] = 1668246626;
        v13[1].m128i_i8[4] = 107;
        *v13 = v14;
        *(_QWORD *)(a2 + 24) += 21LL;
        if ( v11 == v12 )
          goto LABEL_15;
      }
    }
    v16 = sub_16E7EE0(a2, v7, v6);
    v17 = *(_QWORD **)(v16 + 24);
    if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 7u )
    {
      sub_16E7EE0(v16, " (loop: ", 8);
    }
    else
    {
      *v17 = 0x203A706F6F6C2820LL;
      *(_QWORD *)(v16 + 24) += 8LL;
    }
    sub_15537D0(**(_QWORD **)(a1 + 32), a2, 0);
    v22 = *(_WORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v22 <= 1u )
    {
      sub_16E7EE0(a2, ")\n", 2);
    }
    else
    {
      *v22 = 2601;
      *(_QWORD *)(a2 + 24) += 2LL;
    }
    sub_155C2B0(*(_QWORD *)(**(_QWORD **)(a1 + 32) + 56LL), a2, 0);
  }
}
