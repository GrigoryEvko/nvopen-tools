// Function: sub_22A3EC0
// Address: 0x22a3ec0
//
void __fastcall sub_22A3EC0(__int64 a1, __int64 a2)
{
  unsigned __int8 **v2; // rax
  unsigned __int8 **v3; // r12
  unsigned __int8 **v4; // rbx
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  unsigned __int8 *v7; // rdi
  __int64 v8; // rdx
  _BYTE *v9; // rax
  unsigned __int8 **v10; // r9
  unsigned __int8 **v11; // r13
  unsigned __int8 **v12; // r14
  unsigned __int8 *v13; // r10
  void *v14; // rdx
  void *v15; // rdx
  unsigned __int8 *v16; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 16) )
  {
    v2 = *(unsigned __int8 ***)(a1 + 8);
    v3 = &v2[7 * *(unsigned int *)(a1 + 24)];
    if ( v2 != v3 )
    {
      while ( 1 )
      {
        v4 = v2;
        if ( *v2 != (unsigned __int8 *)-4096LL && *v2 != (unsigned __int8 *)-8192LL )
          break;
        v2 += 7;
        if ( v3 == v2 )
          return;
      }
      while ( 1 )
      {
        if ( v3 == v4 )
          return;
        v5 = *(__m128i **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 0x14u )
        {
          sub_CB6200(a2, "  DomFrontier for BB ", 0x15u);
          v7 = *v4;
          if ( !*v4 )
          {
LABEL_33:
            v15 = *(void **)(a2 + 32);
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v15 <= 0xDu )
            {
              sub_CB6200(a2, " <<exit node>>", 0xEu);
              v8 = *(_QWORD *)(a2 + 32);
            }
            else
            {
              qmemcpy(v15, " <<exit node>>", 14);
              v8 = *(_QWORD *)(a2 + 32) + 14LL;
              *(_QWORD *)(a2 + 32) = v8;
            }
            goto LABEL_11;
          }
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4289C50);
          v5[1].m128i_i32[0] = 1111629938;
          v5[1].m128i_i8[4] = 32;
          *v5 = si128;
          *(_QWORD *)(a2 + 32) += 21LL;
          v7 = *v4;
          if ( !*v4 )
            goto LABEL_33;
        }
        sub_A5BF40(v7, a2, 0, 0);
        v8 = *(_QWORD *)(a2 + 32);
LABEL_11:
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 4 )
        {
          sub_CB6200(a2, " is:\t", 5u);
          v9 = *(_BYTE **)(a2 + 32);
        }
        else
        {
          *(_DWORD *)v8 = 980642080;
          *(_BYTE *)(v8 + 4) = 9;
          v9 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 5LL);
          *(_QWORD *)(a2 + 32) = v9;
        }
        v10 = (unsigned __int8 **)v4[5];
        v11 = &v10[*((unsigned int *)v4 + 12)];
        if ( v11 != v10 )
        {
          v12 = (unsigned __int8 **)v4[5];
          do
          {
            while ( 1 )
            {
              v13 = *v12;
              if ( (unsigned __int64)v9 >= *(_QWORD *)(a2 + 24) )
                break;
              *(_QWORD *)(a2 + 32) = v9 + 1;
              *v9 = 32;
              if ( !v13 )
                goto LABEL_20;
LABEL_16:
              sub_A5BF40(v13, a2, 0, 0);
              v9 = *(_BYTE **)(a2 + 32);
LABEL_17:
              if ( v11 == ++v12 )
                goto LABEL_22;
            }
            v16 = *v12;
            sub_CB5D20(a2, 32);
            v13 = v16;
            if ( v16 )
              goto LABEL_16;
LABEL_20:
            v14 = *(void **)(a2 + 32);
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 0xCu )
            {
              sub_CB6200(a2, (unsigned __int8 *)"<<exit node>>", 0xDu);
              v9 = *(_BYTE **)(a2 + 32);
              goto LABEL_17;
            }
            ++v12;
            qmemcpy(v14, "<<exit node>>", 13);
            v9 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 13LL);
            *(_QWORD *)(a2 + 32) = v9;
          }
          while ( v11 != v12 );
        }
LABEL_22:
        if ( (unsigned __int64)v9 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 10);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v9 + 1;
          *v9 = 10;
        }
        v4 += 7;
        if ( v4 == v3 )
          return;
        while ( *v4 == (unsigned __int8 *)-8192LL || *v4 == (unsigned __int8 *)-4096LL )
        {
          v4 += 7;
          if ( v3 == v4 )
            return;
        }
      }
    }
  }
}
