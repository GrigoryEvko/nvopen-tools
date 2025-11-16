// Function: sub_E41190
// Address: 0xe41190
//
char __fastcall sub_E41190(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 v5; // r13
  int v7; // eax
  bool v8; // bl
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // r8
  __m128i *v12; // rdx
  __m128i si128; // xmm0
  char v14; // r8
  unsigned __int8 v15; // r12
  char v16; // al
  char v17; // r8
  int v18; // eax
  __int64 v19; // rdx
  char *v20; // rax
  unsigned __int64 v21; // rdx
  void *v22; // rdx
  int v23; // eax
  char v24; // al
  _BYTE *v25; // rax
  unsigned __int8 *v26; // rax
  __int64 v27; // rdx
  _BYTE *v28; // rax
  int v29; // eax
  __int64 v30; // rdx
  _BYTE *v31; // rax
  unsigned __int8 *v32; // rax
  __int64 v33; // rdx
  char v34; // al
  _BYTE *v35; // rax
  unsigned __int8 v37; // [rsp+8h] [rbp-108h]
  char v38; // [rsp+10h] [rbp-100h]
  char v39; // [rsp+10h] [rbp-100h]
  char v40; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v41; // [rsp+20h] [rbp-F0h] BYREF
  size_t v42; // [rsp+28h] [rbp-E8h]
  _QWORD v43[2]; // [rsp+30h] [rbp-E0h] BYREF
  unsigned __int8 *v44; // [rsp+40h] [rbp-D0h] BYREF
  size_t v45; // [rsp+48h] [rbp-C8h]
  _QWORD v46[2]; // [rsp+50h] [rbp-C0h] BYREF
  unsigned __int8 *v47; // [rsp+60h] [rbp-B0h] BYREF
  size_t v48; // [rsp+68h] [rbp-A8h]
  _QWORD v49[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v50; // [rsp+80h] [rbp-90h]
  __int64 v51; // [rsp+88h] [rbp-88h]
  unsigned __int8 **v52; // [rsp+90h] [rbp-80h]
  __m128i v53; // [rsp+A0h] [rbp-70h] BYREF
  _QWORD v54[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-50h]
  __int64 v56; // [rsp+C8h] [rbp-48h]
  unsigned __int8 **v57; // [rsp+D0h] [rbp-40h]

  v5 = a1;
  if ( (*(_BYTE *)(a2 + 33) & 3) == 2 )
  {
    v8 = sub_B2FC80(a2);
    if ( !v8 )
    {
      if ( a3[11] == 14 && ((v18 = a3[12], v18 == 27) || !v18) )
      {
        v19 = *(_QWORD *)(a1 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v19) <= 8 )
        {
          sub_CB6200(a1, " /EXPORT:", 9u);
        }
        else
        {
          *(_BYTE *)(v19 + 8) = 58;
          *(_QWORD *)v19 = 0x54524F5058452F20LL;
          *(_QWORD *)(a1 + 32) += 9LL;
        }
      }
      else
      {
        v9 = *(_QWORD *)(a1 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v9) <= 8 )
        {
          sub_CB6200(a1, " -export:", 9u);
        }
        else
        {
          *(_BYTE *)(v9 + 8) = 58;
          *(_QWORD *)v9 = 0x74726F7078652D20LL;
          *(_QWORD *)(a1 + 32) += 9LL;
        }
      }
      if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
      {
        v26 = (unsigned __int8 *)sub_BD5D20(a2);
        if ( !(unsigned __int8)sub_E400C0(v26, v27) )
        {
          v28 = *(_BYTE **)(a1 + 32);
          if ( *(_BYTE **)(a1 + 24) == v28 )
          {
            sub_CB6200(a1, (unsigned __int8 *)"\"", 1u);
          }
          else
          {
            *v28 = 34;
            ++*(_QWORD *)(a1 + 32);
          }
          v8 = 1;
        }
      }
      if ( a3[11] == 14 && ((v23 = a3[12], v23 == 1) || v23 == 29) )
      {
        v42 = 0;
        v41 = (unsigned __int8 *)v43;
        v51 = 0x100000000LL;
        v52 = &v41;
        v47 = (unsigned __int8 *)&unk_49DD210;
        LOBYTE(v43[0]) = 0;
        v48 = 0;
        v49[0] = 0;
        v49[1] = 0;
        v50 = 0;
        sub_CB5980((__int64)&v47, 0, 0, 0);
        sub_E409B0(a4, (__int64)&v47, a2);
        if ( v49[0] != v50 )
          sub_CB5AE0((__int64 *)&v47);
        v37 = *v41;
        switch ( *(_DWORD *)(sub_B2F730(a2) + 24) )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
            v24 = 0;
            break;
          case 2:
          case 4:
            v24 = 95;
            break;
          default:
LABEL_89:
            BUG();
        }
        if ( v37 == v24 )
        {
          if ( !v42 )
            goto LABEL_88;
          v53.m128i_i64[0] = (__int64)v54;
          sub_E40120(v53.m128i_i64, v41 + 1, (__int64)&v41[v42]);
          sub_CB6200(a1, (unsigned __int8 *)v53.m128i_i64[0], v53.m128i_u64[1]);
          if ( (_QWORD *)v53.m128i_i64[0] != v54 )
            j_j___libc_free_0(v53.m128i_i64[0], v54[0] + 1LL);
        }
        else
        {
          sub_CB6200(a1, v41, v42);
        }
        v47 = (unsigned __int8 *)&unk_49DD210;
        sub_CB5840((__int64)&v47);
        if ( v41 != (unsigned __int8 *)v43 )
          j_j___libc_free_0(v41, v43[0] + 1LL);
      }
      else
      {
        sub_E409B0(a4, a1, a2);
      }
      if ( a3[8] == 3 && a3[9] == 36 )
      {
        v20 = (char *)sub_BD5D20(a2);
        sub_E406B0(&v53, v20, v21);
        if ( (_BYTE)v55 )
        {
          v22 = *(void **)(a1 + 32);
          if ( *(_QWORD *)(a1 + 24) - (_QWORD)v22 <= 9u )
          {
            a1 = sub_CB6200(a1, ",EXPORTAS,", 0xAu);
          }
          else
          {
            qmemcpy(v22, ",EXPORTAS,", 10);
            *(_QWORD *)(a1 + 32) += 10LL;
          }
          sub_CB6200(a1, (unsigned __int8 *)v53.m128i_i64[0], v53.m128i_u64[1]);
          if ( (_BYTE)v55 )
          {
            LOBYTE(v55) = 0;
            if ( (_QWORD *)v53.m128i_i64[0] != v54 )
              j_j___libc_free_0(v53.m128i_i64[0], v54[0] + 1LL);
          }
        }
      }
      if ( v8 )
      {
        v25 = *(_BYTE **)(v5 + 32);
        if ( *(_BYTE **)(v5 + 24) == v25 )
        {
          sub_CB6200(v5, (unsigned __int8 *)"\"", 1u);
        }
        else
        {
          *v25 = 34;
          ++*(_QWORD *)(v5 + 32);
        }
      }
      if ( *(_BYTE *)(*(_QWORD *)(a2 + 24) + 8LL) != 13 )
      {
        if ( a3[11] == 14 && ((v29 = a3[12], v29 == 27) || !v29) )
        {
          v30 = *(_QWORD *)(v5 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v30) <= 4 )
          {
            sub_CB6200(v5, ",DATA", 5u);
          }
          else
          {
            *(_DWORD *)v30 = 1413563436;
            *(_BYTE *)(v30 + 4) = 65;
            *(_QWORD *)(v5 + 32) += 5LL;
          }
        }
        else
        {
          v10 = *(_QWORD *)(v5 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v10) <= 4 )
          {
            sub_CB6200(v5, ",data", 5u);
          }
          else
          {
            *(_DWORD *)v10 = 1952539692;
            *(_BYTE *)(v10 + 4) = 97;
            *(_QWORD *)(v5 + 32) += 5LL;
          }
        }
      }
    }
  }
  LOBYTE(v7) = *(_BYTE *)(a2 + 32) & 0x30;
  if ( (_BYTE)v7 != 16 )
    return v7;
  LOBYTE(v7) = sub_B2FC80(a2);
  v11 = v7;
  if ( (_BYTE)v7 )
    return v7;
  if ( a3[11] != 14 )
    return v7;
  v7 = a3[12];
  if ( v7 != 29 && v7 != 1 )
    return v7;
  v12 = *(__m128i **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v12 <= 0x11u )
  {
    sub_CB6200(v5, " -exclude-symbols:", 0x12u);
    v11 = 0;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7DB00);
    v12[1].m128i_i16[0] = 14963;
    *v12 = si128;
    *(_QWORD *)(v5 + 32) += 18LL;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v40 = v11;
    v32 = (unsigned __int8 *)sub_BD5D20(a2);
    v34 = sub_E400C0(v32, v33);
    v11 = v40;
    if ( !v34 )
    {
      v35 = *(_BYTE **)(v5 + 32);
      if ( *(_BYTE **)(v5 + 24) == v35 )
      {
        sub_CB6200(v5, (unsigned __int8 *)"\"", 1u);
      }
      else
      {
        *v35 = 34;
        ++*(_QWORD *)(v5 + 32);
      }
      v11 = 1;
    }
  }
  v56 = 0x100000000LL;
  v53.m128i_i64[0] = (__int64)&unk_49DD210;
  v38 = v11;
  v57 = &v44;
  v44 = (unsigned __int8 *)v46;
  v45 = 0;
  LOBYTE(v46[0]) = 0;
  v53.m128i_i64[1] = 0;
  v54[0] = 0;
  v54[1] = 0;
  v55 = 0;
  sub_CB5980((__int64)&v53, 0, 0, 0);
  sub_E409B0(a4, (__int64)&v53, a2);
  v14 = v38;
  if ( v55 != v54[0] )
  {
    sub_CB5AE0(v53.m128i_i64);
    v14 = v38;
  }
  v39 = v14;
  v15 = *v44;
  switch ( *(_DWORD *)(sub_B2F730(a2) + 24) )
  {
    case 0:
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
      v16 = 0;
      goto LABEL_28;
    case 2:
    case 4:
      v16 = 95;
LABEL_28:
      if ( v15 == v16 )
      {
        if ( v45 )
        {
          v47 = (unsigned __int8 *)v49;
          sub_E40120((__int64 *)&v47, v44 + 1, (__int64)&v44[v45]);
          sub_CB6200(v5, v47, v48);
          v17 = v39;
          if ( v47 != (unsigned __int8 *)v49 )
          {
            j_j___libc_free_0(v47, v49[0] + 1LL);
            v17 = v39;
          }
          goto LABEL_30;
        }
LABEL_88:
        sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
      }
      sub_CB6200(v5, v44, v45);
      v17 = v39;
LABEL_30:
      if ( v17 )
      {
        v31 = *(_BYTE **)(v5 + 32);
        if ( *(_BYTE **)(v5 + 24) == v31 )
        {
          sub_CB6200(v5, (unsigned __int8 *)"\"", 1u);
        }
        else
        {
          *v31 = 34;
          ++*(_QWORD *)(v5 + 32);
        }
      }
      v53.m128i_i64[0] = (__int64)&unk_49DD210;
      LOBYTE(v7) = (unsigned __int8)sub_CB5840((__int64)&v53);
      if ( v44 != (unsigned __int8 *)v46 )
        LOBYTE(v7) = j_j___libc_free_0(v44, v46[0] + 1LL);
      return v7;
    default:
      goto LABEL_89;
  }
}
