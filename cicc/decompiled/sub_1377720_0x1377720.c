// Function: sub_1377720
// Address: 0x1377720
//
__int64 __fastcall sub_1377720(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  size_t v12; // rdx
  _DWORD *v13; // rdi
  const char *v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  size_t v17; // rdx
  __m128i *v18; // rdi
  const char *v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rbx
  const char *v23; // r13
  size_t v24; // rax
  _DWORD *v25; // rcx
  size_t v26; // rdx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  char *v33; // rcx
  const char *v34; // r13
  unsigned int v35; // ecx
  unsigned int v36; // ecx
  unsigned int v37; // eax
  __int64 v38; // rsi
  size_t v39; // [rsp+0h] [rbp-50h]
  size_t v40; // [rsp+0h] [rbp-50h]
  int v41; // [rsp+Ch] [rbp-44h]
  int v42[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = sub_13774B0(a1, a3, a4);
  v9 = *(_QWORD *)(a2 + 24);
  v41 = v8;
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v9) <= 4 )
  {
    v10 = sub_16E7EE0(a2, "edge ", 5);
  }
  else
  {
    *(_DWORD *)v9 = 1701274725;
    v10 = a2;
    *(_BYTE *)(v9 + 4) = 32;
    *(_QWORD *)(a2 + 24) += 5LL;
  }
  v11 = sub_1649960(a3);
  v13 = *(_DWORD **)(v10 + 24);
  v14 = (const char *)v11;
  v15 = *(_QWORD *)(v10 + 16) - (_QWORD)v13;
  if ( v15 < v12 )
  {
    v31 = sub_16E7EE0(v10, v14);
    v13 = *(_DWORD **)(v31 + 24);
    v10 = v31;
    v15 = *(_QWORD *)(v31 + 16) - (_QWORD)v13;
  }
  else if ( v12 )
  {
    v40 = v12;
    memcpy(v13, v14, v12);
    v13 = (_DWORD *)(v40 + *(_QWORD *)(v10 + 24));
    v29 = *(_QWORD *)(v10 + 16) - (_QWORD)v13;
    *(_QWORD *)(v10 + 24) = v13;
    if ( v29 > 3 )
      goto LABEL_6;
    goto LABEL_19;
  }
  if ( v15 > 3 )
  {
LABEL_6:
    *v13 = 540945696;
    *(_QWORD *)(v10 + 24) += 4LL;
    goto LABEL_7;
  }
LABEL_19:
  v10 = sub_16E7EE0(v10, " -> ", 4);
LABEL_7:
  v16 = sub_1649960(a4);
  v18 = *(__m128i **)(v10 + 24);
  v19 = (const char *)v16;
  v20 = *(_QWORD *)(v10 + 16) - (_QWORD)v18;
  if ( v20 < v17 )
  {
    v30 = sub_16E7EE0(v10, v19);
    v18 = *(__m128i **)(v30 + 24);
    v10 = v30;
    v20 = *(_QWORD *)(v30 + 16) - (_QWORD)v18;
  }
  else if ( v17 )
  {
    v39 = v17;
    memcpy(v18, v19, v17);
    v18 = (__m128i *)(v39 + *(_QWORD *)(v10 + 24));
    v28 = *(_QWORD *)(v10 + 16) - (_QWORD)v18;
    *(_QWORD *)(v10 + 24) = v18;
    if ( v28 > 0xF )
      goto LABEL_10;
    goto LABEL_17;
  }
  if ( v20 > 0xF )
  {
LABEL_10:
    *v18 = _mm_load_si128((const __m128i *)&xmmword_3F8CFA0);
    *(_QWORD *)(v10 + 24) += 16LL;
    goto LABEL_11;
  }
LABEL_17:
  v10 = sub_16E7EE0(v10, " probability is ", 16);
LABEL_11:
  v42[0] = v41;
  v21 = a3;
  v22 = sub_16AF620(v42, v10);
  v23 = " [HOT edge]\n";
  if ( !sub_13776D0(a1, v21, a4) )
    v23 = "\n";
  v24 = strlen(v23);
  v25 = *(_DWORD **)(v22 + 24);
  v26 = v24;
  if ( *(_QWORD *)(v22 + 16) - (_QWORD)v25 >= v24 )
  {
    if ( (unsigned int)v24 < 8 )
    {
      if ( (v24 & 4) != 0 )
      {
        *v25 = *(_DWORD *)v23;
        *(_DWORD *)((char *)v25 + (unsigned int)v24 - 4) = *(_DWORD *)&v23[(unsigned int)v24 - 4];
      }
      else if ( (_DWORD)v24 )
      {
        *(_BYTE *)v25 = *v23;
        if ( (v24 & 2) != 0 )
          *(_WORD *)((char *)v25 + (unsigned int)v24 - 2) = *(_WORD *)&v23[(unsigned int)v24 - 2];
      }
    }
    else
    {
      v32 = (unsigned __int64)(v25 + 2) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v25 = *(_QWORD *)v23;
      *(_QWORD *)((char *)v25 + (unsigned int)v24 - 8) = *(_QWORD *)&v23[(unsigned int)v24 - 8];
      v33 = (char *)v25 - v32;
      v34 = (const char *)(v23 - v33);
      v35 = (v24 + (_DWORD)v33) & 0xFFFFFFF8;
      if ( v35 >= 8 )
      {
        v36 = v35 & 0xFFFFFFF8;
        v37 = 0;
        do
        {
          v38 = v37;
          v37 += 8;
          *(_QWORD *)(v32 + v38) = *(_QWORD *)&v34[v38];
        }
        while ( v37 < v36 );
      }
    }
    *(_QWORD *)(v22 + 24) += v26;
  }
  else
  {
    sub_16E7EE0(v22, v23);
  }
  return a2;
}
