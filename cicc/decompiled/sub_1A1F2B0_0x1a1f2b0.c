// Function: sub_1A1F2B0
// Address: 0x1a1f2b0
//
unsigned __int64 __fastcall sub_1A1F2B0(
        __int64 a1,
        __int64 a2,
        __int64 **a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int64 *a6,
        unsigned __int64 a7)
{
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  int v12; // r9d
  __int64 v13; // rdx
  unsigned __int64 result; // rax
  __int64 i; // rbx
  int v16; // r8d
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // r9d
  __int32 v21; // r8d
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // r13
  char v25; // al
  __int64 v26; // rdx
  __int64 v27; // rsi
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  int v30; // r9d
  unsigned int v31; // edx
  __int64 v32; // rbx
  int v33; // r8d
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // r9d
  __int32 v38; // r8d
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r13
  char v43; // al
  __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // eax
  int v49; // [rsp-8h] [rbp-F8h]
  __int64 v50; // [rsp+10h] [rbp-E0h]
  int v51; // [rsp+18h] [rbp-D8h]
  __int64 v52; // [rsp+18h] [rbp-D8h]
  int v54; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v55; // [rsp+28h] [rbp-C8h]
  int v56; // [rsp+34h] [rbp-BCh]
  int v57; // [rsp+34h] [rbp-BCh]
  __m128i v61; // [rsp+60h] [rbp-90h] BYREF
  __int16 v62; // [rsp+70h] [rbp-80h]
  __m128i v63; // [rsp+80h] [rbp-70h] BYREF
  __int16 v64; // [rsp+90h] [rbp-60h]
  __m128i v65[5]; // [rsp+A0h] [rbp-50h] BYREF

  v9 = sub_157EB90(*(_QWORD *)(a1 + 8));
  v10 = sub_1632FA0(v9);
  if ( *(_BYTE *)(a2 + 8) == 14 )
  {
    v11 = sub_127FA20(v10, *(_QWORD *)(a2 + 24));
    v13 = *(_QWORD *)(a2 + 32);
    result = (unsigned __int64)(v11 + 7) >> 3;
    v51 = result;
    if ( (_DWORD)v13 )
    {
      v56 = 0;
      result = *(unsigned int *)(a1 + 112);
      v50 = (unsigned int)v13;
      for ( i = 0; i != v50; ++i )
      {
        v16 = i;
        if ( (unsigned int)result >= *(_DWORD *)(a1 + 116) )
        {
          sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 4, i, v12);
          result = *(unsigned int *)(a1 + 112);
          v16 = i;
        }
        v54 = v16;
        *(_DWORD *)(*(_QWORD *)(a1 + 104) + 4 * result) = v16;
        v17 = *(_QWORD **)(a1 + 24);
        ++*(_DWORD *)(a1 + 112);
        v18 = sub_1643350(v17);
        v19 = sub_159C470(v18, i, 0);
        v21 = v54;
        v22 = v19;
        v23 = *(unsigned int *)(a1 + 144);
        if ( (unsigned int)v23 >= *(_DWORD *)(a1 + 148) )
        {
          sub_16CD150(a1 + 136, (const void *)(a1 + 152), 0, 8, v54, v20);
          v23 = *(unsigned int *)(a1 + 144);
          v21 = v54;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v23) = v22;
        v63.m128i_i32[0] = v21;
        ++*(_DWORD *)(a1 + 144);
        v24 = (v56 | a5) & (unsigned __int64)-(__int64)(v56 | a5);
        v64 = 265;
        v25 = *(_BYTE *)(a4 + 16);
        if ( v25 )
        {
          if ( v25 == 1 )
          {
            v61.m128i_i64[0] = (__int64)".";
            v62 = 259;
          }
          else
          {
            if ( *(_BYTE *)(a4 + 17) == 1 )
            {
              v26 = *(_QWORD *)a4;
            }
            else
            {
              v26 = a4;
              v25 = 2;
            }
            v61.m128i_i64[0] = v26;
            v61.m128i_i64[1] = (__int64)".";
            LOBYTE(v62) = v25;
            HIBYTE(v62) = 3;
          }
        }
        else
        {
          v62 = 256;
        }
        sub_14EC200(v65, &v61, &v63);
        v27 = *(_QWORD *)(a2 + 24);
        v28 = *(unsigned __int8 *)(v27 + 8);
        if ( (unsigned __int8)v28 <= 0x10u && (v29 = 100990, _bittest64(&v29, v28)) )
        {
          sub_1A1EF10(a1, v27, a3, v65, v24, a6);
        }
        else
        {
          sub_1A1F2B0(a1, v27, (_DWORD)a3, (unsigned int)v65, v24, (_DWORD)a6, a7);
          v12 = v49;
        }
        --*(_DWORD *)(a1 + 144);
        result = (unsigned int)(*(_DWORD *)(a1 + 112) - 1);
        *(_DWORD *)(a1 + 112) = result;
        if ( a7 <= *a6 )
          break;
        v56 += v51;
      }
    }
  }
  else
  {
    result = sub_15A9930(v10, a2);
    v31 = *(_DWORD *)(a2 + 12);
    v55 = result;
    if ( v31 )
    {
      v52 = v31;
      v32 = 0;
      v33 = 0;
      result = *(unsigned int *)(a1 + 112);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 116) )
        goto LABEL_37;
      while ( 1 )
      {
        v57 = v33;
        *(_DWORD *)(*(_QWORD *)(a1 + 104) + 4 * result) = v33;
        v34 = *(_QWORD **)(a1 + 24);
        ++*(_DWORD *)(a1 + 112);
        v35 = sub_1643350(v34);
        v36 = sub_159C470(v35, v32, 0);
        v38 = v57;
        v39 = v36;
        v40 = *(unsigned int *)(a1 + 144);
        if ( (unsigned int)v40 >= *(_DWORD *)(a1 + 148) )
        {
          sub_16CD150(a1 + 136, (const void *)(a1 + 152), 0, 8, v57, v37);
          v40 = *(unsigned int *)(a1 + 144);
          v38 = v57;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v40) = v39;
        ++*(_DWORD *)(a1 + 144);
        v41 = *(_QWORD *)(v55 + 8 * v32 + 16) | a5;
        v64 = 265;
        v63.m128i_i32[0] = v38;
        v42 = v41 & -v41;
        v43 = *(_BYTE *)(a4 + 16);
        if ( v43 )
        {
          if ( v43 == 1 )
          {
            v61.m128i_i64[0] = (__int64)".";
            v62 = 259;
          }
          else
          {
            if ( *(_BYTE *)(a4 + 17) == 1 )
            {
              v44 = *(_QWORD *)a4;
            }
            else
            {
              v44 = a4;
              v43 = 2;
            }
            v61.m128i_i64[0] = v44;
            v61.m128i_i64[1] = (__int64)".";
            LOBYTE(v62) = v43;
            HIBYTE(v62) = 3;
          }
        }
        else
        {
          v62 = 256;
        }
        sub_14EC200(v65, &v61, &v63);
        v45 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v32);
        v46 = *(unsigned __int8 *)(v45 + 8);
        if ( (unsigned __int8)v46 <= 0x10u && (v47 = 100990, _bittest64(&v47, v46)) )
          sub_1A1EF10(a1, v45, a3, v65, v42, a6);
        else
          sub_1A1F2B0(a1, v45, (_DWORD)a3, (unsigned int)v65, v42, (_DWORD)a6, a7);
        v48 = *(_DWORD *)(a1 + 112);
        --*(_DWORD *)(a1 + 144);
        result = (unsigned int)(v48 - 1);
        *(_DWORD *)(a1 + 112) = result;
        if ( a7 <= *a6 )
          break;
        if ( v52 == ++v32 )
          break;
        v33 = v32;
        if ( (unsigned int)result >= *(_DWORD *)(a1 + 116) )
        {
LABEL_37:
          sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 4, v33, v30);
          result = *(unsigned int *)(a1 + 112);
          v33 = v32;
        }
      }
    }
  }
  return result;
}
