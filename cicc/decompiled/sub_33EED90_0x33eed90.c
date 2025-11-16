// Function: sub_33EED90
// Address: 0x33eed90
//
__int64 __fastcall sub_33EED90(__int64 a1, const char *a2, unsigned int a3, __int64 a4)
{
  __int64 *v4; // r15
  size_t v5; // r12
  int v7; // eax
  unsigned int v8; // r8d
  _QWORD *v9; // rcx
  __int64 v10; // r14
  __int64 result; // rax
  __int64 v12; // rax
  unsigned int v13; // r8d
  _QWORD *v14; // rcx
  _QWORD *v15; // r14
  __int64 *v16; // rax
  __int64 *v17; // rax
  __m128i *v18; // rax
  int v19; // edx
  unsigned __int64 v20; // r12
  __m128i *v21; // r15
  int v22; // r8d
  unsigned __int8 *v23; // rsi
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v29; // [rsp+10h] [rbp-50h]
  unsigned int v30; // [rsp+18h] [rbp-48h]
  int v31; // [rsp+18h] [rbp-48h]
  unsigned __int8 *v32; // [rsp+28h] [rbp-38h] BYREF

  v4 = (__int64 *)(a1 + 920);
  v5 = 0;
  if ( a2 )
    v5 = strlen(a2);
  v7 = sub_C92610();
  v8 = sub_C92740((__int64)v4, a2, v5, v7);
  v9 = (_QWORD *)(*(_QWORD *)(a1 + 920) + 8LL * v8);
  v10 = *v9;
  if ( *v9 )
  {
    if ( v10 != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 936);
  }
  v29 = v9;
  v30 = v8;
  v12 = sub_C7D670(v5 + 17, 8);
  v13 = v30;
  v14 = v29;
  v15 = (_QWORD *)v12;
  if ( v5 )
  {
    memcpy((void *)(v12 + 16), a2, v5);
    v13 = v30;
    v14 = v29;
  }
  *((_BYTE *)v15 + v5 + 16) = 0;
  *v15 = v5;
  v15[1] = 0;
  *v14 = v15;
  ++*(_DWORD *)(a1 + 932);
  v16 = (__int64 *)(*(_QWORD *)(a1 + 920) + 8LL * (unsigned int)sub_C929D0(v4, v13));
  v10 = *v16;
  if ( *v16 != -8 && v10 )
  {
LABEL_5:
    result = *(_QWORD *)(v10 + 8);
    if ( result )
      return result;
LABEL_16:
    v18 = sub_33ED250(a1, a3, a4);
    v20 = *(_QWORD *)(a1 + 416);
    v21 = v18;
    v22 = v19;
    if ( v20 )
    {
      *(_QWORD *)(a1 + 416) = *(_QWORD *)v20;
    }
    else
    {
      v24 = *(_QWORD *)(a1 + 424);
      *(_QWORD *)(a1 + 504) += 120LL;
      v25 = (v24 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_QWORD *)(a1 + 432) >= v25 + 120 && v24 )
      {
        *(_QWORD *)(a1 + 424) = v25 + 120;
        if ( !v25 )
        {
LABEL_21:
          *(_QWORD *)(v10 + 8) = v20;
          sub_33CC420(a1, v20);
          return *(_QWORD *)(v10 + 8);
        }
        v20 = (v24 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v31 = v19;
        v26 = sub_9D1E70(a1 + 424, 120, 120, 3);
        v22 = v31;
        v20 = v26;
      }
    }
    v32 = 0;
    *(_QWORD *)v20 = 0;
    v23 = v32;
    *(_QWORD *)(v20 + 8) = 0;
    *(_QWORD *)(v20 + 16) = 0;
    *(_QWORD *)(v20 + 24) = 18;
    *(_WORD *)(v20 + 34) = -1;
    *(_DWORD *)(v20 + 36) = -1;
    *(_QWORD *)(v20 + 40) = 0;
    *(_QWORD *)(v20 + 48) = v21;
    *(_QWORD *)(v20 + 56) = 0;
    *(_DWORD *)(v20 + 64) = 0;
    *(_DWORD *)(v20 + 68) = v22;
    *(_DWORD *)(v20 + 72) = 0;
    *(_QWORD *)(v20 + 80) = v23;
    if ( v23 )
      sub_B976B0((__int64)&v32, v23, v20 + 80);
    *(_QWORD *)(v20 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v20 + 32) = 0;
    *(_QWORD *)(v20 + 96) = a2;
    *(_DWORD *)(v20 + 104) = 0;
    goto LABEL_21;
  }
  v17 = v16 + 1;
  do
  {
    do
      v10 = *v17++;
    while ( v10 == -8 );
  }
  while ( !v10 );
  result = *(_QWORD *)(v10 + 8);
  if ( !result )
    goto LABEL_16;
  return result;
}
