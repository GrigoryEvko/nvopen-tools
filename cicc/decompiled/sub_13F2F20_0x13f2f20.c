// Function: sub_13F2F20
// Address: 0x13f2f20
//
void __fastcall sub_13F2F20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned int i; // r15d
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rbx
  char v12; // dl
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r15
  unsigned __int8 v16; // al
  __int64 v17; // rdi
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 *v20; // rsi
  unsigned int v21; // r8d
  __int64 *v22; // rcx
  __int64 v23; // r8
  __m128i *v24; // rdx
  __m128i si128; // xmm0
  void *v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rdi
  _BYTE *v30; // rax
  int v31; // [rsp+10h] [rbp-140h]
  __int64 v32; // [rsp+10h] [rbp-140h]
  __int64 v33; // [rsp+18h] [rbp-138h] BYREF
  __int64 *v34; // [rsp+20h] [rbp-130h] BYREF
  __int64 v35; // [rsp+28h] [rbp-128h]
  __int64 *v36; // [rsp+30h] [rbp-120h]
  __int64 v37; // [rsp+38h] [rbp-118h]
  int v38; // [rsp+40h] [rbp-110h] BYREF
  __int64 v39; // [rsp+48h] [rbp-108h]
  unsigned int v40; // [rsp+50h] [rbp-100h]
  __int64 v41; // [rsp+58h] [rbp-F8h]
  unsigned int v42; // [rsp+60h] [rbp-F0h]
  __int64 v43; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE *v44; // [rsp+78h] [rbp-D8h]
  _BYTE *v45; // [rsp+80h] [rbp-D0h]
  __int64 v46; // [rsp+88h] [rbp-C8h]
  int v47; // [rsp+90h] [rbp-C0h]
  _BYTE v48[184]; // [rsp+98h] [rbp-B8h] BYREF

  v44 = v48;
  v4 = *(_QWORD *)(a2 + 40);
  v45 = v48;
  v36 = &v33;
  v33 = a2;
  v35 = a1;
  v43 = 0;
  v46 = 16;
  v47 = 0;
  v34 = &v43;
  v37 = a3;
  sub_13F2AA0((__int64 *)&v34, v4);
  v5 = sub_157EBA0(v4);
  if ( v5 )
  {
    v31 = sub_15F4D60(v5);
    v6 = sub_157EBA0(v4);
    if ( v31 )
    {
      for ( i = 0; i != v31; ++i )
      {
        while ( 1 )
        {
          v8 = sub_15F4DF0(v6, i);
          if ( (unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 16), v4, v8, v9, v10) )
            break;
          if ( v31 == ++i )
            goto LABEL_7;
        }
        sub_13F2AA0((__int64 *)&v34, v8);
      }
    }
  }
LABEL_7:
  v11 = *(_QWORD *)(v33 + 8);
  if ( v11 )
  {
    while ( 1 )
    {
      v15 = sub_1648700(v11);
      v16 = *(_BYTE *)(v15 + 16);
      if ( v16 <= 0x17u
        || v16 == 77 && !(unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 16), v4, *(_QWORD *)(v15 + 40), v13, v14) )
      {
        goto LABEL_10;
      }
      v17 = (__int64)v34;
      v18 = *(_QWORD *)(v15 + 40);
      v19 = (__int64 *)v34[1];
      if ( (__int64 *)v34[2] == v19 )
      {
        v20 = &v19[*((unsigned int *)v34 + 7)];
        v21 = *((_DWORD *)v34 + 7);
        if ( v19 != v20 )
        {
          v22 = 0;
          while ( v18 != *v19 )
          {
            if ( *v19 == -2 )
              v22 = v19;
            if ( v20 == ++v19 )
            {
              if ( !v22 )
                goto LABEL_46;
              *v22 = v18;
              --*(_DWORD *)(v17 + 32);
              ++*(_QWORD *)v17;
              goto LABEL_22;
            }
          }
          goto LABEL_10;
        }
LABEL_46:
        if ( v21 < *((_DWORD *)v34 + 6) )
          break;
      }
      sub_16CCBA0(v34, *(_QWORD *)(v15 + 40));
      if ( v12 )
      {
LABEL_22:
        sub_13F2700(&v38, *(_QWORD *)(v35 + 8), *v36, v18, 0);
        v23 = v37;
        v24 = *(__m128i **)(v37 + 24);
        if ( *(_QWORD *)(v37 + 16) - (_QWORD)v24 <= 0x12u )
        {
          v23 = sub_16E7EE0(v37, "; LatticeVal for: '", 19);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_428A450);
          v24[1].m128i_i8[2] = 39;
          v24[1].m128i_i16[0] = 8250;
          *v24 = si128;
          *(_QWORD *)(v23 + 24) += 19LL;
        }
        v32 = v23;
        sub_155C2B0(*v36, v23, 0);
        v26 = *(void **)(v32 + 24);
        if ( *(_QWORD *)(v32 + 16) - (_QWORD)v26 <= 9u )
        {
          sub_16E7EE0(v32, "' in BB: '", 10);
        }
        else
        {
          qmemcpy(v26, "' in BB: '", 10);
          *(_QWORD *)(v32 + 24) += 10LL;
        }
        sub_15537D0(v18, v37, 0);
        v27 = v37;
        v28 = *(_QWORD *)(v37 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v37 + 16) - v28) <= 5 )
        {
          v27 = sub_16E7EE0(v37, "' is: ", 6);
        }
        else
        {
          *(_DWORD *)v28 = 1936269351;
          *(_WORD *)(v28 + 4) = 8250;
          *(_QWORD *)(v27 + 24) += 6LL;
        }
        v29 = sub_14A8A60(v27, &v38);
        v30 = *(_BYTE **)(v29 + 24);
        if ( *(_BYTE **)(v29 + 16) == v30 )
        {
          sub_16E7EE0(v29, "\n", 1);
        }
        else
        {
          *v30 = 10;
          ++*(_QWORD *)(v29 + 24);
        }
        if ( v38 != 3 )
          goto LABEL_10;
        if ( v42 > 0x40 && v41 )
          j_j___libc_free_0_0(v41);
        if ( v40 <= 0x40 || !v39 )
          goto LABEL_10;
        j_j___libc_free_0_0(v39);
        v11 = *(_QWORD *)(v11 + 8);
        if ( !v11 )
          goto LABEL_37;
      }
      else
      {
LABEL_10:
        v11 = *(_QWORD *)(v11 + 8);
        if ( !v11 )
          goto LABEL_37;
      }
    }
    *((_DWORD *)v34 + 7) = v21 + 1;
    *v20 = v18;
    ++*(_QWORD *)v17;
    goto LABEL_22;
  }
LABEL_37:
  if ( v45 != v44 )
    _libc_free((unsigned __int64)v45);
}
