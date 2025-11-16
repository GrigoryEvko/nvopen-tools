// Function: sub_FF0830
// Address: 0xff0830
//
__int64 __fastcall sub_FF0830(__int64 a1, __int64 a2, unsigned __int8 *a3, unsigned __int8 *a4)
{
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rdx
  int v9; // r15d
  __int64 v10; // rax
  _DWORD *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r15
  char *v20; // r13
  size_t v21; // rax
  _DWORD *v22; // rcx
  size_t v23; // rdx
  unsigned __int64 v25; // rdi
  char *v26; // rcx
  char *v27; // r13
  unsigned int v28; // ecx
  unsigned int v29; // ecx
  unsigned int v30; // eax
  __int64 v31; // rsi
  int v32[13]; // [rsp+Ch] [rbp-34h] BYREF

  v6 = a2;
  v7 = sub_FF0430(a1, (__int64)a3, (__int64)a4);
  v8 = *(_QWORD *)(a2 + 32);
  v9 = v7;
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 4 )
  {
    sub_CB6200(a2, "edge ", 5u);
  }
  else
  {
    *(_DWORD *)v8 = 1701274725;
    *(_BYTE *)(v8 + 4) = 32;
    *(_QWORD *)(a2 + 32) += 5LL;
  }
  v10 = sub_AA4B30((__int64)a3);
  sub_A5BF40(a3, a2, 0, v10);
  v11 = *(_DWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 3u )
  {
    sub_CB6200(a2, (unsigned __int8 *)" -> ", 4u);
  }
  else
  {
    *v11 = 540945696;
    *(_QWORD *)(a2 + 32) += 4LL;
  }
  v12 = sub_AA4B30((__int64)a4);
  sub_A5BF40(a4, a2, 0, v12);
  v16 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v16 <= 0xFu )
  {
    a2 = sub_CB6200(a2, " probability is ", 0x10u);
  }
  else
  {
    *v16 = _mm_load_si128((const __m128i *)&xmmword_3F8CFA0);
    *(_QWORD *)(a2 + 32) += 16LL;
  }
  v32[0] = v9;
  v17 = sub_F02CC0(v32, a2, (__int64)v16, v13, v14, v15);
  v18 = (__int64)a3;
  v19 = v17;
  v20 = " [HOT edge]\n";
  if ( !sub_FF06D0(a1, v18, (__int64)a4) )
    v20 = "\n";
  v21 = strlen(v20);
  v22 = *(_DWORD **)(v19 + 32);
  v23 = v21;
  if ( *(_QWORD *)(v19 + 24) - (_QWORD)v22 >= v21 )
  {
    if ( (unsigned int)v21 < 8 )
    {
      if ( (v21 & 4) != 0 )
      {
        *v22 = *(_DWORD *)v20;
        *(_DWORD *)((char *)v22 + (unsigned int)v21 - 4) = *(_DWORD *)&v20[(unsigned int)v21 - 4];
      }
      else if ( (_DWORD)v21 )
      {
        *(_BYTE *)v22 = *v20;
        if ( (v21 & 2) != 0 )
          *(_WORD *)((char *)v22 + (unsigned int)v21 - 2) = *(_WORD *)&v20[(unsigned int)v21 - 2];
      }
    }
    else
    {
      v25 = (unsigned __int64)(v22 + 2) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v22 = *(_QWORD *)v20;
      *(_QWORD *)((char *)v22 + (unsigned int)v21 - 8) = *(_QWORD *)&v20[(unsigned int)v21 - 8];
      v26 = (char *)v22 - v25;
      v27 = (char *)(v20 - v26);
      v28 = (v21 + (_DWORD)v26) & 0xFFFFFFF8;
      if ( v28 >= 8 )
      {
        v29 = v28 & 0xFFFFFFF8;
        v30 = 0;
        do
        {
          v31 = v30;
          v30 += 8;
          *(_QWORD *)(v25 + v31) = *(_QWORD *)&v27[v31];
        }
        while ( v30 < v29 );
      }
    }
    *(_QWORD *)(v19 + 32) += v23;
  }
  else
  {
    sub_CB6200(v19, (unsigned __int8 *)v20, v21);
  }
  return v6;
}
