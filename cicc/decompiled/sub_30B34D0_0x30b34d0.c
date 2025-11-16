// Function: sub_30B34D0
// Address: 0x30b34d0
//
__int64 *__fastcall sub_30B34D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __m128i *v5; // rdx
  _QWORD *v6; // rdi
  __int64 v7; // rax
  _WORD *v8; // rdx
  int v9; // eax
  __int64 *v10; // rcx
  __int64 *v11; // rbx
  __int64 *v12; // r14
  unsigned __int64 *v13; // rax
  __m128i *v15; // rdx
  __m128i *v16; // rdx
  __m128i si128; // xmm0
  _QWORD *v18; // rcx
  int v19; // r13d
  _QWORD *v20; // rbx
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  __int64 *v23; // [rsp+8h] [rbp-D8h]
  _QWORD *v24; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v25[2]; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE v26[16]; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int8 *v27[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v28; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v29[3]; // [rsp+70h] [rbp-70h] BYREF
  __m128i *v30; // [rsp+88h] [rbp-58h]
  __m128i *v31; // [rsp+90h] [rbp-50h]
  __int64 v32; // [rsp+98h] [rbp-48h]
  unsigned __int64 *v33; // [rsp+A0h] [rbp-40h]

  v25[0] = (unsigned __int64)v26;
  v32 = 0x100000000LL;
  v33 = v25;
  v25[1] = 0;
  v29[0] = &unk_49DD210;
  v26[0] = 0;
  v29[1] = 0;
  v29[2] = 0;
  v30 = 0;
  v31 = 0;
  sub_CB5980((__int64)v29, 0, 0, 0);
  v5 = v31;
  if ( (unsigned __int64)((char *)v30 - (char *)v31) <= 5 )
  {
    v6 = (_QWORD *)sub_CB6200((__int64)v29, "<kind:", 6u);
  }
  else
  {
    v31->m128i_i32[0] = 1852402492;
    v6 = v29;
    v5->m128i_i16[2] = 14948;
    v31 = (__m128i *)((char *)v31 + 6);
  }
  v7 = sub_30B0C30((__int64)v6, *(_DWORD *)(a2 + 56));
  v8 = *(_WORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    sub_CB6200(v7, (unsigned __int8 *)">\n", 2u);
  }
  else
  {
    *v8 = 2622;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  v9 = *(_DWORD *)(a2 + 56);
  if ( (unsigned int)(v9 - 1) > 1 )
  {
    if ( v9 == 3 )
    {
      v16 = v31;
      if ( (unsigned __int64)((char *)v30 - (char *)v31) <= 0x22 )
      {
        sub_CB6200((__int64)v29, "--- start of nodes in pi-block ---\n", 0x23u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44CBAD0);
        v31[2].m128i_i8[2] = 10;
        v16[2].m128i_i16[0] = 11565;
        *v16 = si128;
        v16[1] = _mm_load_si128((const __m128i *)&xmmword_44CBAE0);
        v31 = (__m128i *)((char *)v31 + 35);
      }
      v18 = *(_QWORD **)(a2 + 64);
      v24 = &v18[*(unsigned int *)(a2 + 72)];
      if ( v24 != v18 )
      {
        v23 = a1;
        v19 = 0;
        v20 = *(_QWORD **)(a2 + 64);
        do
        {
          sub_30B34D0(v27, *v20, a3);
          sub_CB6200((__int64)v29, v27[0], (size_t)v27[1]);
          if ( (__int64 *)v27[0] != &v28 )
            j_j___libc_free_0((unsigned __int64)v27[0]);
          if ( ++v19 != *(_DWORD *)(a2 + 72) )
          {
            if ( v30 == v31 )
            {
              sub_CB6200((__int64)v29, (unsigned __int8 *)"\n", 1u);
            }
            else
            {
              v31->m128i_i8[0] = 10;
              v31 = (__m128i *)((char *)v31 + 1);
            }
          }
          ++v20;
        }
        while ( v24 != v20 );
        a1 = v23;
      }
      v21 = v31;
      if ( (unsigned __int64)((char *)v30 - (char *)v31) <= 0x20 )
      {
        sub_CB6200((__int64)v29, "--- end of nodes in pi-block ---\n", 0x21u);
      }
      else
      {
        v22 = _mm_load_si128((const __m128i *)&xmmword_44CBAF0);
        v31[2].m128i_i8[0] = 10;
        *v21 = v22;
        v21[1] = _mm_load_si128((const __m128i *)&xmmword_44CBB00);
        v31 = (__m128i *)((char *)v31 + 33);
      }
    }
    else
    {
      if ( v9 != 4 )
        BUG();
      v15 = v31;
      if ( (unsigned __int64)((char *)v30 - (char *)v31) <= 4 )
      {
        sub_CB6200((__int64)v29, "root\n", 5u);
      }
      else
      {
        v31->m128i_i32[0] = 1953460082;
        v15->m128i_i8[4] = 10;
        v31 = (__m128i *)((char *)v31 + 5);
      }
    }
  }
  else
  {
    v10 = *(__int64 **)(a2 + 64);
    v11 = &v10[*(unsigned int *)(a2 + 72)];
    if ( v10 != v11 )
    {
      v12 = *(__int64 **)(a2 + 64);
      do
      {
        while ( 1 )
        {
          sub_A69870(*v12, v29, 0);
          if ( v30 == v31 )
            break;
          ++v12;
          v31->m128i_i8[0] = 10;
          v31 = (__m128i *)((char *)v31 + 1);
          if ( v11 == v12 )
            goto LABEL_11;
        }
        ++v12;
        sub_CB6200((__int64)v29, (unsigned __int8 *)"\n", 1u);
      }
      while ( v11 != v12 );
    }
  }
LABEL_11:
  v13 = v33;
  *a1 = (__int64)(a1 + 2);
  sub_30B3180(a1, (_BYTE *)*v13, *v13 + v13[1]);
  v29[0] = &unk_49DD210;
  sub_CB5840((__int64)v29);
  if ( (_BYTE *)v25[0] != v26 )
    j_j___libc_free_0(v25[0]);
  return a1;
}
