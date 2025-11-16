// Function: sub_23C6990
// Address: 0x23c6990
//
void __fastcall sub_23C6990(__int64 a1, _BYTE **a2)
{
  __int128 *v2; // rax
  __int128 *v3; // r12
  unsigned int v4; // eax
  unsigned __int8 *v5; // rdi
  __int64 *v6; // rdi
  _QWORD *v7; // rax
  void *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __m128i *v14; // rdx
  __m128i si128; // xmm0
  unsigned __int8 *v16; // [rsp+0h] [rbp-40h] BYREF
  size_t v17; // [rsp+8h] [rbp-38h]
  _BYTE v18[48]; // [rsp+10h] [rbp-30h] BYREF

  v2 = sub_23C67A0();
  v3 = v2;
  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock((pthread_mutex_t *)v2);
    if ( v4 )
      sub_4264C5(v4);
  }
  v5 = *a2;
  v18[0] = 0;
  v16 = v18;
  v17 = 0;
  if ( (_UNKNOWN *)sub_3157040(v5, &v16) == &unk_5034368 )
  {
    v7 = sub_CB72A0();
    v8 = (void *)v7[4];
    v9 = (__int64)v7;
    if ( v7[3] - (_QWORD)v8 <= 0xEu )
    {
      v9 = sub_CB6200((__int64)v7, "Error opening '", 0xFu);
    }
    else
    {
      qmemcpy(v8, "Error opening '", 15);
      v7[4] += 15LL;
    }
    v10 = sub_CB6200(v9, *a2, (size_t)a2[1]);
    v11 = *(_QWORD *)(v10 + 32);
    v12 = v10;
    if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v11) <= 2 )
    {
      v12 = sub_CB6200(v10, "': ", 3u);
    }
    else
    {
      *(_BYTE *)(v11 + 2) = 32;
      *(_WORD *)v11 = 14887;
      *(_QWORD *)(v10 + 32) += 3LL;
    }
    v13 = sub_CB6200(v12, v16, v17);
    v14 = *(__m128i **)(v13 + 32);
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 0x19u )
    {
      sub_CB6200(v13, "\n  -load request ignored.\n", 0x1Au);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42AEF60);
      qmemcpy(&v14[1], " ignored.\n", 10);
      *v14 = si128;
      *(_QWORD *)(v13 + 32) += 26LL;
    }
  }
  else
  {
    v6 = (__int64 *)*((_QWORD *)v3 + 7);
    if ( v6 == *((__int64 **)v3 + 8) )
    {
      sub_8FD760((__m128i **)v3 + 6, *((const __m128i **)v3 + 7), (__int64)a2);
    }
    else
    {
      if ( v6 )
      {
        *v6 = (__int64)(v6 + 2);
        sub_23C6860(v6, *a2, (__int64)&a2[1][(_QWORD)*a2]);
        v6 = (__int64 *)*((_QWORD *)v3 + 7);
      }
      *((_QWORD *)v3 + 7) = v6 + 4;
    }
  }
  if ( v16 != v18 )
    j_j___libc_free_0((unsigned __int64)v16);
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)v3);
}
