// Function: sub_15B81B0
// Address: 0x15b81b0
//
__int64 __fastcall sub_15B81B0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // rcx
  __int64 v10; // rdx
  char v11; // dl
  int v12; // eax
  int v13; // ebx
  int v14; // eax
  __int64 v15; // rsi
  int v16; // r8d
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rcx
  __int64 v20; // rdx
  int v21; // [rsp+Ch] [rbp-74h] BYREF
  __int64 v22; // [rsp+10h] [rbp-70h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h] BYREF
  __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  __int64 v25; // [rsp+28h] [rbp-58h] BYREF
  __m128i v26; // [rsp+30h] [rbp-50h]
  char v27; // [rsp+40h] [rbp-40h]
  __int64 v28; // [rsp+48h] [rbp-38h]
  char v29; // [rsp+50h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = *(unsigned int *)(*a2 + 8);
    v24 = *(_QWORD *)(*a2 - 8 * v9);
    v10 = *(_QWORD *)(v6 + 8 * (1 - v9));
    v27 = *(_BYTE *)(v6 + 40);
    v25 = v10;
    v11 = *(_BYTE *)(v6 + 56);
    if ( v27 )
    {
      v29 = *(_BYTE *)(v6 + 56);
      v26 = _mm_loadu_si128((const __m128i *)(v6 + 24));
      if ( v11 )
      {
        v28 = *(_QWORD *)(v6 + 48);
        v23 = v28;
      }
      else
      {
        v23 = 0;
      }
      v22 = v26.m128i_i64[1];
      v12 = v26.m128i_i32[0];
    }
    else
    {
      v29 = *(_BYTE *)(v6 + 56);
      if ( v11 )
      {
        v28 = *(_QWORD *)(v6 + 48);
        v23 = v28;
      }
      else
      {
        v23 = 0;
      }
      v22 = 0;
      v12 = 0;
    }
    v21 = v12;
    v13 = v4 - 1;
    v14 = sub_15B5960(&v24, &v25, &v21, &v22, &v23);
    v15 = *a2;
    v16 = 1;
    v17 = 0;
    v18 = v13 & v14;
    v19 = (_QWORD *)(v7 + 8LL * v18);
    v20 = *v19;
    if ( *a2 == *v19 )
    {
LABEL_19:
      *a3 = v19;
      return 1;
    }
    else
    {
      while ( v20 != -8 )
      {
        if ( v20 != -16 || v17 )
          v19 = v17;
        v18 = v13 & (v16 + v18);
        v20 = *(_QWORD *)(v7 + 8LL * v18);
        if ( v20 == v15 )
        {
          v19 = (_QWORD *)(v7 + 8LL * v18);
          goto LABEL_19;
        }
        ++v16;
        v17 = v19;
        v19 = (_QWORD *)(v7 + 8LL * v18);
      }
      if ( !v17 )
        v17 = v19;
      *a3 = v17;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
