// Function: sub_385DDE0
// Address: 0x385dde0
//
__int64 __fastcall sub_385DDE0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 result; // rax
  unsigned int v6; // r12d
  unsigned int i; // ecx
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rax
  __m128i *v17; // rdx
  __int64 v18; // rdi
  __m128i si128; // xmm0
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // r15d
  __int64 v23; // r13
  __int64 j; // rax
  __int64 v25; // r12
  _BYTE *v26; // rax
  __int64 v27; // rax
  void *v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned int v32; // ebx
  __int64 v33; // r15
  __int64 k; // rax
  __int64 v35; // r14
  _BYTE *v36; // rax
  __int64 v37; // rax
  __int64 *v38; // [rsp+0h] [rbp-60h]
  __int64 v39; // [rsp+8h] [rbp-58h]
  unsigned int v41; // [rsp+14h] [rbp-4Ch]
  __int64 *v42; // [rsp+18h] [rbp-48h]
  unsigned int v43; // [rsp+20h] [rbp-40h]
  unsigned int v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+20h] [rbp-40h]

  result = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  v42 = *(__int64 **)a3;
  v38 = (__int64 *)result;
  if ( *(_QWORD *)a3 != result )
  {
    v6 = a4 + 2;
    for ( i = 0; ; i = v41 )
    {
      v43 = i;
      v8 = *v42;
      v9 = v42[1];
      v10 = sub_16E8750(a2, a4);
      v11 = v43;
      v12 = *(_QWORD *)(v10 + 24);
      v13 = v10;
      if ( (unsigned __int64)(*(_QWORD *)(v10 + 16) - v12) <= 5 )
      {
        v37 = sub_16E7EE0(v10, "Check ", 6u);
        v11 = v43;
        v13 = v37;
      }
      else
      {
        *(_DWORD *)v12 = 1667590211;
        *(_WORD *)(v12 + 4) = 8299;
        *(_QWORD *)(v10 + 24) += 6LL;
      }
      v41 = v11 + 1;
      v14 = sub_16E7A90(v13, v11);
      v15 = *(_WORD **)(v14 + 24);
      if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 1u )
      {
        sub_16E7EE0(v14, ":\n", 2u);
      }
      else
      {
        *v15 = 2618;
        *(_QWORD *)(v14 + 24) += 2LL;
      }
      v16 = sub_16E8750(a2, v6);
      v17 = *(__m128i **)(v16 + 24);
      v18 = v16;
      if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 0x10u )
      {
        v18 = sub_16E7EE0(v16, "Comparing group (", 0x11u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F717C0);
        v17[1].m128i_i8[0] = 40;
        *v17 = si128;
        *(_QWORD *)(v16 + 24) += 17LL;
      }
      v20 = sub_16E7B40(v18, *v42);
      v21 = *(_QWORD *)(v20 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v20 + 16) - v21) <= 2 )
      {
        sub_16E7EE0(v20, "):\n", 3u);
      }
      else
      {
        *(_BYTE *)(v21 + 2) = 10;
        *(_WORD *)v21 = 14889;
        *(_QWORD *)(v20 + 24) += 3LL;
      }
      v22 = 0;
      if ( *(_DWORD *)(v8 + 32) )
      {
        v39 = v9;
        v23 = 0;
        v44 = v6;
        for ( j = sub_16E8750(a2, v6); ; j = sub_16E8750(a2, v44) )
        {
          v25 = j;
          sub_155C2B0(
            *(_QWORD *)(*(_QWORD *)(a1 + 8)
                      + ((unsigned __int64)*(unsigned int *)(*(_QWORD *)(v8 + 24) + 4 * v23) << 6)
                      + 16),
            j,
            0);
          v26 = *(_BYTE **)(v25 + 24);
          if ( *(_BYTE **)(v25 + 16) == v26 )
          {
            sub_16E7EE0(v25, "\n", 1u);
            v23 = ++v22;
            if ( v22 >= *(_DWORD *)(v8 + 32) )
            {
LABEL_17:
              v9 = v39;
              v6 = v44;
              break;
            }
          }
          else
          {
            v23 = v22 + 1;
            *v26 = 10;
            v22 = v23;
            ++*(_QWORD *)(v25 + 24);
            if ( (unsigned int)v23 >= *(_DWORD *)(v8 + 32) )
              goto LABEL_17;
          }
        }
      }
      v27 = sub_16E8750(a2, v6);
      v28 = *(void **)(v27 + 24);
      v29 = v27;
      if ( *(_QWORD *)(v27 + 16) - (_QWORD)v28 <= 0xEu )
      {
        v29 = sub_16E7EE0(v27, "Against group (", 0xFu);
      }
      else
      {
        qmemcpy(v28, "Against group (", 15);
        *(_QWORD *)(v27 + 24) += 15LL;
      }
      v30 = sub_16E7B40(v29, v42[1]);
      v31 = *(_QWORD *)(v30 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v30 + 16) - v31) <= 2 )
      {
        sub_16E7EE0(v30, "):\n", 3u);
      }
      else
      {
        *(_BYTE *)(v31 + 2) = 10;
        *(_WORD *)v31 = 14889;
        *(_QWORD *)(v30 + 24) += 3LL;
      }
      if ( *(_DWORD *)(v9 + 32) )
      {
        v45 = a2;
        v32 = 0;
        v33 = 0;
        for ( k = sub_16E8750(v45, v6); ; k = sub_16E8750(v45, v6) )
        {
          v35 = k;
          sub_155C2B0(
            *(_QWORD *)(*(_QWORD *)(a1 + 8)
                      + ((unsigned __int64)*(unsigned int *)(*(_QWORD *)(v9 + 24) + 4 * v33) << 6)
                      + 16),
            k,
            0);
          v36 = *(_BYTE **)(v35 + 24);
          if ( *(_BYTE **)(v35 + 16) == v36 )
          {
            sub_16E7EE0(v35, "\n", 1u);
            v33 = ++v32;
            if ( v32 >= *(_DWORD *)(v9 + 32) )
            {
LABEL_28:
              a2 = v45;
              break;
            }
          }
          else
          {
            v33 = v32 + 1;
            *v36 = 10;
            v32 = v33;
            ++*(_QWORD *)(v35 + 24);
            if ( (unsigned int)v33 >= *(_DWORD *)(v9 + 32) )
              goto LABEL_28;
          }
        }
      }
      v42 += 2;
      result = (__int64)v42;
      if ( v38 == v42 )
        break;
    }
  }
  return result;
}
