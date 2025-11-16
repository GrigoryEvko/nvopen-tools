// Function: sub_37DE580
// Address: 0x37de580
//
int *__fastcall sub_37DE580(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  int *result; // rax
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rsi
  int v12; // r11d
  __int64 v13; // r9
  unsigned int v14; // eax
  int *v15; // r13
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rax
  char v20; // si
  __int64 v21; // r8
  __int64 v22; // rdi
  __int32 v23; // edx
  __m128i *v24; // rax
  __m128i v25; // xmm0
  char v26; // dl
  __int64 v27; // r9
  int v28; // esi
  unsigned int v29; // r8d
  int *v30; // rax
  int v31; // r10d
  unsigned int v32; // esi
  unsigned int v33; // eax
  int v34; // edi
  unsigned int v35; // r8d
  int v36; // edx
  int v37; // r11d
  int *v38; // rcx
  __int64 v39; // [rsp+8h] [rbp-E8h]
  __int64 v40; // [rsp+18h] [rbp-D8h]
  __int64 *v41; // [rsp+20h] [rbp-D0h]
  int v42; // [rsp+44h] [rbp-ACh] BYREF
  int *v43; // [rsp+48h] [rbp-A8h] BYREF
  __m128i v44; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+60h] [rbp-90h]
  char v46; // [rsp+68h] [rbp-88h]
  __int64 v47; // [rsp+70h] [rbp-80h]
  __m128i v48[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v49; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v50; // [rsp+A8h] [rbp-48h]
  int v51; // [rsp+B8h] [rbp-38h]

  v6 = *(_QWORD *)a2;
  result = *(int **)(a1 + 832);
  if ( *(_BYTE *)(a2 + 24) )
  {
    v8 = *(_QWORD *)(a2 + 8);
    v9 = *(_QWORD *)(a2 + 16);
  }
  else
  {
    v8 = qword_4F81350[0];
    v9 = qword_4F81350[1];
  }
  v10 = (unsigned int)result[6];
  v11 = *((_QWORD *)result + 1);
  if ( (_DWORD)v10 )
  {
    v12 = 1;
    for ( result = (int *)(((_DWORD)v10 - 1)
                         & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                          * ((unsigned int)(unsigned __int16)v9
                                           | ((_DWORD)v8 << 16)
                                           | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
                          ^ (484763065 * ((unsigned __int16)v9 | ((_DWORD)v8 << 16)))));
          ;
          result = (int *)(((_DWORD)v10 - 1) & v14) )
    {
      v13 = v11 + 56LL * (unsigned int)result;
      if ( v6 == *(_QWORD *)v13 && v8 == *(_QWORD *)(v13 + 8) && v9 == *(_QWORD *)(v13 + 16) )
        break;
      if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -1 && *(_QWORD *)(v13 + 16) == -1 )
        return result;
      v14 = v12 + (_DWORD)result;
      ++v12;
    }
    result = (int *)(v11 + 56 * v10);
    if ( (int *)v13 != result )
    {
      v15 = *(int **)(v13 + 24);
      result = &v15[4 * *(unsigned int *)(v13 + 32)];
      v41 = (__int64 *)result;
      if ( result != v15 )
      {
        v40 = a1 + 8;
        v39 = a1 + 680;
        v16 = a1;
        v17 = *(__int64 **)(v13 + 24);
        v18 = v16;
        while ( 1 )
        {
          v19 = *v17;
          if ( *v17 != qword_4F81350[0] || (v20 = 0, v17[1] != qword_4F81350[1]) )
            v20 = 1;
          v45 = v17[1];
          v21 = *(_QWORD *)(a2 + 32);
          v46 = v20;
          v22 = *(_QWORD *)v18;
          v44.m128i_i64[0] = v6;
          v44.m128i_i64[1] = v19;
          v47 = v21;
          v42 = sub_37CCE30(v22, &v44, a3);
          v23 = dword_5051178[0];
          v24 = v48;
          do
          {
            v24->m128i_i32[0] = v23;
            v24 = (__m128i *)((char *)v24 + 4);
          }
          while ( &v49 != (__int64 *)v24 );
          v25 = _mm_loadu_si128((const __m128i *)(v18 + 840));
          v49 = 0;
          v51 = 0;
          v50 = v25;
          sub_37DE250(v40, &v42, v48);
          v26 = *(_BYTE *)(v18 + 688) & 1;
          if ( v26 )
          {
            v27 = v18 + 696;
            v28 = 7;
          }
          else
          {
            v32 = *(_DWORD *)(v18 + 704);
            v27 = *(_QWORD *)(v18 + 696);
            if ( !v32 )
            {
              v33 = *(_DWORD *)(v18 + 688);
              ++*(_QWORD *)(v18 + 680);
              v43 = 0;
              v34 = (v33 >> 1) + 1;
LABEL_31:
              v35 = 3 * v32;
              goto LABEL_32;
            }
            v28 = v32 - 1;
          }
          v29 = v28 & (37 * v42);
          v30 = (int *)(v27 + 16LL * v29);
          v31 = *v30;
          if ( *v30 == v42 )
            goto LABEL_24;
          v37 = 1;
          v38 = 0;
          while ( v31 != -1 )
          {
            if ( v31 == -2 && !v38 )
              v38 = v30;
            v29 = v28 & (v37 + v29);
            v30 = (int *)(v27 + 16LL * v29);
            v31 = *v30;
            if ( v42 == *v30 )
              goto LABEL_24;
            ++v37;
          }
          v35 = 24;
          v32 = 8;
          if ( !v38 )
            v38 = v30;
          v33 = *(_DWORD *)(v18 + 688);
          ++*(_QWORD *)(v18 + 680);
          v43 = v38;
          v34 = (v33 >> 1) + 1;
          if ( !v26 )
          {
            v32 = *(_DWORD *)(v18 + 704);
            goto LABEL_31;
          }
LABEL_32:
          if ( 4 * v34 >= v35 )
          {
            v32 *= 2;
LABEL_38:
            sub_37C5F80(v39, v32);
            sub_37BDA60(v39, &v42, &v43);
            v33 = *(_DWORD *)(v18 + 688);
            goto LABEL_34;
          }
          if ( v32 - *(_DWORD *)(v18 + 692) - v34 <= v32 >> 3 )
            goto LABEL_38;
LABEL_34:
          *(_DWORD *)(v18 + 688) = (2 * (v33 >> 1) + 2) | v33 & 1;
          v30 = v43;
          if ( *v43 != -1 )
            --*(_DWORD *)(v18 + 692);
          v36 = v42;
          *((_QWORD *)v30 + 1) = 0;
          *v30 = v36;
LABEL_24:
          result = v30 + 2;
          v17 += 2;
          *(_QWORD *)result = a3;
          if ( v41 == v17 )
            return result;
          v6 = *(_QWORD *)a2;
        }
      }
    }
  }
  return result;
}
