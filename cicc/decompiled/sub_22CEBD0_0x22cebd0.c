// Function: sub_22CEBD0
// Address: 0x22cebd0
//
void __fastcall sub_22CEBD0(__int64 a1, __int64 a2, unsigned __int8 **a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 **v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  unsigned int i; // r15d
  unsigned __int8 *v15; // r12
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 *v18; // rdi
  unsigned __int8 *v19; // r12
  unsigned __int8 **v20; // rax
  __int64 v21; // r8
  __m128i *v22; // rdx
  __m128i si128; // xmm0
  void *v24; // rdx
  __int64 v25; // rdi
  unsigned __int8 *v26; // rdx
  __int64 v27; // rdi
  _BYTE *v28; // rax
  int v29; // [rsp+10h] [rbp-130h]
  __int64 v30; // [rsp+10h] [rbp-130h]
  __int64 v31; // [rsp+18h] [rbp-128h] BYREF
  __int64 *v32; // [rsp+20h] [rbp-120h] BYREF
  __int64 v33; // [rsp+28h] [rbp-118h]
  __int64 *v34; // [rsp+30h] [rbp-110h]
  unsigned __int8 **v35; // [rsp+38h] [rbp-108h]
  char v36[8]; // [rsp+40h] [rbp-100h] BYREF
  unsigned __int64 v37; // [rsp+48h] [rbp-F8h]
  unsigned int v38; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v39; // [rsp+58h] [rbp-E8h]
  unsigned int v40; // [rsp+60h] [rbp-E0h]
  __int64 v41; // [rsp+70h] [rbp-D0h] BYREF
  _BYTE *v42; // [rsp+78h] [rbp-C8h]
  __int64 v43; // [rsp+80h] [rbp-C0h]
  int v44; // [rsp+88h] [rbp-B8h]
  char v45; // [rsp+8Ch] [rbp-B4h]
  _BYTE v46[176]; // [rsp+90h] [rbp-B0h] BYREF

  v32 = &v41;
  v7 = *(_QWORD *)(a2 + 40);
  v34 = &v31;
  v31 = a2;
  v33 = a1;
  v35 = a3;
  v41 = 0;
  v42 = v46;
  v43 = 16;
  v44 = 0;
  v45 = 1;
  sub_22CE390((__int64)&v32, (unsigned __int8 *)v7, a3, (__int64)v46, a5, a6);
  v11 = (unsigned __int8 **)(v7 + 48);
  v12 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v12 != v7 + 48 )
  {
    if ( !v12 )
      BUG();
    v13 = v12 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 <= 0xA )
    {
      v29 = sub_B46E30(v13);
      if ( v29 )
      {
        for ( i = 0; i != v29; ++i )
        {
          while ( 1 )
          {
            v15 = (unsigned __int8 *)sub_B46EC0(v13, i);
            if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 16), v7, (__int64)v15) )
              break;
            if ( v29 == ++i )
              goto LABEL_9;
          }
          sub_22CE390((__int64)&v32, v15, v11, v8, v9, v10);
        }
      }
    }
  }
LABEL_9:
  v16 = *(_QWORD *)(v31 + 16);
  if ( v16 )
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v16 + 24);
      if ( *(_BYTE *)v17 <= 0x1Cu )
        goto LABEL_17;
      if ( *(_BYTE *)v17 != 84 || (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 16), v7, *(_QWORD *)(v17 + 40)) )
      {
        v18 = v32;
        v19 = *(unsigned __int8 **)(v17 + 40);
        if ( *((_BYTE *)v32 + 28) )
        {
          v20 = (unsigned __int8 **)v32[1];
          v8 = *((unsigned int *)v32 + 5);
          v11 = &v20[v8];
          if ( v20 != v11 )
          {
            while ( v19 != *v20 )
            {
              if ( v11 == ++v20 )
                goto LABEL_40;
            }
            goto LABEL_17;
          }
LABEL_40:
          if ( (unsigned int)v8 < *((_DWORD *)v32 + 4) )
          {
            *((_DWORD *)v32 + 5) = v8 + 1;
            *v11 = v19;
            ++*v18;
            goto LABEL_21;
          }
        }
        sub_C8CC70((__int64)v32, (__int64)v19, (__int64)v11, v8, v9, v10);
        if ( (_BYTE)v11 )
        {
LABEL_21:
          sub_22CDEF0((__int64)v36, *(_QWORD *)(v33 + 8), *v34, (__int64)v19, 0);
          v21 = (__int64)v35;
          v22 = (__m128i *)v35[4];
          if ( (unsigned __int64)(v35[3] - (unsigned __int8 *)v22) <= 0x12 )
          {
            v21 = sub_CB6200((__int64)v35, "; LatticeVal for: '", 0x13u);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_428A450);
            v22[1].m128i_i8[2] = 39;
            v22[1].m128i_i16[0] = 8250;
            *v22 = si128;
            *(_QWORD *)(v21 + 32) += 19LL;
          }
          v30 = v21;
          sub_A69870(*v34, (_BYTE *)v21, 0);
          v24 = *(void **)(v30 + 32);
          if ( *(_QWORD *)(v30 + 24) - (_QWORD)v24 <= 9u )
          {
            sub_CB6200(v30, "' in BB: '", 0xAu);
          }
          else
          {
            qmemcpy(v24, "' in BB: '", 10);
            *(_QWORD *)(v30 + 32) += 10LL;
          }
          sub_A5BF40(v19, (__int64)v35, 0, 0);
          v25 = (__int64)v35;
          v26 = v35[4];
          if ( (unsigned __int64)(v35[3] - v26) <= 5 )
          {
            v25 = sub_CB6200((__int64)v35, "' is: ", 6u);
          }
          else
          {
            *(_DWORD *)v26 = 1936269351;
            *((_WORD *)v26 + 2) = 8250;
            *(_QWORD *)(v25 + 32) += 6LL;
          }
          v27 = sub_22EAFB0(v25, v36);
          v28 = *(_BYTE **)(v27 + 32);
          if ( *(_BYTE **)(v27 + 24) == v28 )
          {
            sub_CB6200(v27, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v28 = 10;
            ++*(_QWORD *)(v27 + 32);
          }
          if ( (unsigned int)(unsigned __int8)v36[0] - 4 > 1 )
            goto LABEL_17;
          if ( v40 > 0x40 && v39 )
            j_j___libc_free_0_0(v39);
          if ( v38 <= 0x40 || !v37 )
            goto LABEL_17;
          j_j___libc_free_0_0(v37);
          v16 = *(_QWORD *)(v16 + 8);
          if ( !v16 )
            break;
        }
        else
        {
LABEL_17:
          v16 = *(_QWORD *)(v16 + 8);
          if ( !v16 )
            break;
        }
      }
      else
      {
        v16 = *(_QWORD *)(v16 + 8);
        if ( !v16 )
          break;
      }
    }
  }
  if ( !v45 )
    _libc_free((unsigned __int64)v42);
}
