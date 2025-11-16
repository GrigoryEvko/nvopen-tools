// Function: sub_2E44300
// Address: 0x2e44300
//
__int64 __fastcall sub_2E44300(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rsi
  _QWORD *v10; // rdi
  __int64 v11; // rdx
  _DWORD *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i *v16; // rdx
  __int64 v17; // rsi
  char *v18; // r13
  size_t v19; // rax
  size_t v20; // rdx
  _DWORD *v21; // rcx
  unsigned __int64 v23; // rdi
  char *v24; // rcx
  char *v25; // r13
  unsigned int v26; // ecx
  unsigned int v27; // ecx
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+10h] [rbp-90h]
  int v34; // [rsp+2Ch] [rbp-74h] BYREF
  _QWORD v35[2]; // [rsp+30h] [rbp-70h] BYREF
  void (__fastcall *v36)(_QWORD *, _QWORD *, __int64); // [rsp+40h] [rbp-60h]
  void (__fastcall *v37)(_QWORD *, __int64); // [rsp+48h] [rbp-58h]
  _QWORD v38[2]; // [rsp+50h] [rbp-50h] BYREF
  void (__fastcall *v39)(_QWORD *, _QWORD *, __int64); // [rsp+60h] [rbp-40h]
  void (__fastcall *v40)(_QWORD *, __int64); // [rsp+68h] [rbp-38h]

  v6 = sub_2E441D0(a1, a3, a4);
  v7 = *(_QWORD *)(a2 + 32);
  v31 = v6;
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v7) <= 4 )
  {
    v8 = sub_CB6200(a2, "edge ", 5u);
  }
  else
  {
    *(_DWORD *)v7 = 1701274725;
    v8 = a2;
    *(_BYTE *)(v7 + 4) = 32;
    *(_QWORD *)(a2 + 32) += 5LL;
  }
  v9 = a3;
  v10 = v35;
  sub_2E31000(v35, a3);
  if ( !v36
    || ((v37(v35, v8), v12 = *(_DWORD **)(v8 + 32), *(_QWORD *)(v8 + 24) - (_QWORD)v12 <= 3u)
      ? (v8 = sub_CB6200(v8, (unsigned __int8 *)" -> ", 4u))
      : (*v12 = 540945696, *(_QWORD *)(v8 + 32) += 4LL),
        v9 = a4,
        v10 = v38,
        sub_2E31000(v38, a4),
        !v39) )
  {
    sub_4263D6(v10, v9, v11);
  }
  v40(v38, v8);
  v16 = *(__m128i **)(v8 + 32);
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v16 <= 0xFu )
  {
    v8 = sub_CB6200(v8, " probability is ", 0x10u);
  }
  else
  {
    *v16 = _mm_load_si128((const __m128i *)&xmmword_3F8CFA0);
    *(_QWORD *)(v8 + 32) += 16LL;
  }
  v34 = v31;
  v17 = a3;
  v32 = sub_F02CC0(&v34, v8, (__int64)v16, v13, v14, v15);
  v18 = " [HOT edge]\n";
  if ( !sub_2E442A0(a1, v17, a4) )
    v18 = "\n";
  v19 = strlen(v18);
  v20 = v19;
  v21 = *(_DWORD **)(v32 + 32);
  if ( *(_QWORD *)(v32 + 24) - (_QWORD)v21 >= v19 )
  {
    if ( (unsigned int)v19 < 8 )
    {
      if ( (v19 & 4) != 0 )
      {
        *v21 = *(_DWORD *)v18;
        *(_DWORD *)((char *)v21 + (unsigned int)v19 - 4) = *(_DWORD *)&v18[(unsigned int)v19 - 4];
      }
      else if ( (_DWORD)v19 )
      {
        *(_BYTE *)v21 = *v18;
        if ( (v19 & 2) != 0 )
          *(_WORD *)((char *)v21 + (unsigned int)v19 - 2) = *(_WORD *)&v18[(unsigned int)v19 - 2];
      }
    }
    else
    {
      v23 = (unsigned __int64)(v21 + 2) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v21 = *(_QWORD *)v18;
      *(_QWORD *)((char *)v21 + (unsigned int)v19 - 8) = *(_QWORD *)&v18[(unsigned int)v19 - 8];
      v24 = (char *)v21 - v23;
      v25 = (char *)(v18 - v24);
      v26 = (v19 + (_DWORD)v24) & 0xFFFFFFF8;
      if ( v26 >= 8 )
      {
        v27 = v26 & 0xFFFFFFF8;
        v28 = 0;
        do
        {
          v29 = v28;
          v28 += 8;
          *(_QWORD *)(v23 + v29) = *(_QWORD *)&v25[v29];
        }
        while ( v28 < v27 );
      }
    }
    *(_QWORD *)(v32 + 32) += v20;
  }
  else
  {
    sub_CB6200(v32, (unsigned __int8 *)v18, v19);
  }
  if ( v39 )
    v39(v38, v38, 3);
  if ( v36 )
    v36(v35, v35, 3);
  return a2;
}
