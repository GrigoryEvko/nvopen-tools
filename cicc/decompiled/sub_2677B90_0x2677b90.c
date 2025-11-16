// Function: sub_2677B90
// Address: 0x2677b90
//
__m128i *__fastcall sub_2677B90(__m128i *a1, __int64 a2)
{
  __m128i *v5; // rax
  const char *v6; // rsi
  __int64 v7; // rax
  unsigned int v8; // edi
  unsigned __int64 *v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // rdx
  int v13; // r8d
  unsigned __int64 v14; // rcx
  int v15; // eax
  unsigned __int64 v16; // rdi
  __int64 v17; // [rsp+8h] [rbp-88h]
  unsigned __int64 v18; // [rsp+10h] [rbp-80h]
  int v19; // [rsp+1Ch] [rbp-74h]
  unsigned __int64 v20[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v21[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v22[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v23[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( !*(_BYTE *)(a2 + 97) )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    strcpy(a1[1].m128i_i8, "<invalid>");
    a1->m128i_i64[1] = 9;
    return a1;
  }
  v22[0] = 18;
  v20[0] = (unsigned __int64)v21;
  v5 = (__m128i *)sub_22409D0((__int64)v20, (unsigned __int64 *)v22, 0);
  v6 = "none";
  v20[0] = (unsigned __int64)v5;
  v21[0] = v22[0];
  *v5 = _mm_load_si128((const __m128i *)&xmmword_438FCD0);
  v5[1].m128i_i16[0] = 8250;
  v20[1] = v22[0];
  *(_BYTE *)(v20[0] + v22[0]) = 0;
  if ( *(_BYTE *)(a2 + 112) )
  {
    v7 = *(_QWORD *)(a2 + 104);
    if ( !v7 )
    {
      sub_26712D0(v22, "nullptr");
      sub_2677B10(a1, (__int64)v20, (unsigned __int64 *)v22);
      sub_2240A30((unsigned __int64 *)v22);
      goto LABEL_23;
    }
    if ( *(_BYTE *)v7 == 17 )
    {
      v8 = *(_DWORD *)(v7 + 32);
      v9 = *(unsigned __int64 **)(v7 + 24);
      if ( v8 > 0x40 )
      {
        v11 = *v9;
        v10 = *v9;
      }
      else
      {
        if ( !v8 )
        {
          v17 = 0;
          v11 = 0;
          v13 = 1;
          goto LABEL_21;
        }
        v10 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v8)) >> (64 - (unsigned __int8)v8);
        v11 = v10;
      }
      if ( v10 >= 0 )
      {
        v17 = 0;
        v8 = 0;
      }
      else
      {
        v17 = 1;
        v11 = -(__int64)v11;
        v8 = 1;
      }
      if ( v11 <= 9 )
      {
        v13 = 1;
      }
      else if ( v11 <= 0x63 )
      {
        v13 = 2;
      }
      else if ( v11 <= 0x3E7 )
      {
        v13 = 3;
      }
      else if ( v11 <= 0x270F )
      {
        v13 = 4;
      }
      else
      {
        v12 = v11;
        v13 = 1;
        while ( 1 )
        {
          v14 = v12;
          v15 = v13;
          v13 += 4;
          v12 /= 0x2710u;
          if ( v14 <= 0x1869F )
            break;
          if ( v14 <= 0xF423F )
          {
            v13 = v15 + 5;
            break;
          }
          if ( v14 <= (unsigned __int64)&loc_98967F )
          {
            v13 = v15 + 6;
            break;
          }
          if ( v14 <= 0x5F5E0FF )
          {
            v13 = v15 + 7;
            break;
          }
        }
      }
LABEL_21:
      v18 = v11;
      v19 = v13;
      v22[0] = (__int64)v23;
      sub_2240A50(v22, v13 + v8, 45);
      sub_1249540((_BYTE *)(v22[0] + v17), v19, v18);
      sub_2677B10(a1, (__int64)v20, (unsigned __int64 *)v22);
      v16 = v22[0];
      if ( (_QWORD *)v22[0] == v23 )
        goto LABEL_23;
      goto LABEL_22;
    }
    v6 = "unknown";
  }
  sub_26712D0(v22, v6);
  sub_2677B10(a1, (__int64)v20, (unsigned __int64 *)v22);
  v16 = v22[0];
  if ( (_QWORD *)v22[0] != v23 )
LABEL_22:
    j_j___libc_free_0(v16);
LABEL_23:
  if ( (_QWORD *)v20[0] != v21 )
    j_j___libc_free_0(v20[0]);
  return a1;
}
