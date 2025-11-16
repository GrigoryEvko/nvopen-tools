// Function: sub_DF2CB0
// Address: 0xdf2cb0
//
__int64 __fastcall sub_DF2CB0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rdx
  char v9; // cl
  __int64 v10; // r15
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // r14
  int v14; // eax
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // r15
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // [rsp+0h] [rbp-80h]
  __int64 *v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h] BYREF
  int v30; // [rsp+18h] [rbp-68h]
  void *v31; // [rsp+20h] [rbp-60h] BYREF
  __int64 v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h]
  __int64 v34; // [rsp+38h] [rbp-48h]
  __int64 i; // [rsp+40h] [rbp-40h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  sub_C7D6A0(0, 0, 8);
  v4 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v4;
  if ( (_DWORD)v4 )
  {
    v17 = sub_C7D670(24 * v4, 8);
    v18 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 + 8) = v17;
    v19 = v17;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    if ( *(_DWORD *)(a1 + 24) )
    {
      v20 = 0;
      v21 = 24LL * *(unsigned int *)(a1 + 24);
      do
      {
        v22 = *(_QWORD *)(v18 + v20);
        *(_QWORD *)(v19 + v20) = v22;
        if ( v22 != -8192 && v22 != -4096 )
          *(__m128i *)(v19 + v20 + 8) = _mm_loadu_si128((const __m128i *)(v18 + v20 + 8));
        v20 += 24;
      }
      while ( v21 != v20 );
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  *(_DWORD *)(a1 + 56) = 128;
  v5 = a1 + 32;
  *(_QWORD *)(a1 + 32) = 0;
  v6 = (_QWORD *)sub_C7D670(6144, 8);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = v6;
  v7 = *(unsigned int *)(a1 + 56);
  v32 = 2;
  v33 = 0;
  v34 = -4096;
  v31 = &unk_49DDFA0;
  v8 = &v6[6 * v7];
  for ( i = 0; v8 != v6; v6 += 6 )
  {
    if ( v6 )
    {
      v9 = v32;
      v6[2] = 0;
      v6[3] = -4096;
      *v6 = &unk_49DDFA0;
      v6[1] = v9 & 6;
      v6[4] = i;
    }
  }
  *(_BYTE *)(a1 + 96) = 0;
  v10 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a1 + 120) = *(_QWORD *)(a2 + 120);
  v11 = *(_QWORD *)(a2 + 128);
  *(_QWORD *)(a1 + 112) = v10;
  v27 = *(unsigned int *)(v11 + 48);
  v28 = *(__int64 **)(v11 + 40);
  v12 = (_QWORD *)sub_22077B0(184);
  v13 = v12;
  if ( v12 )
    sub_D9AF00(v12, v28, v27, v10);
  *(_QWORD *)(a1 + 128) = v13;
  v14 = *(_DWORD *)(a2 + 136);
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 136) = v14;
  v15 = *(_QWORD *)(a2 + 144);
  *(_BYTE *)(a1 + 164) = 0;
  *(_QWORD *)(a1 + 144) = v15;
  result = *(unsigned int *)(a2 + 48);
  if ( (_DWORD)result )
  {
    result = *(unsigned int *)(a2 + 56);
    v23 = *(_QWORD *)(a2 + 40);
    v24 = v23 + 48 * result;
    if ( v23 != v24 )
    {
      while ( 1 )
      {
        result = *(_QWORD *)(v23 + 24);
        if ( result != -4096 && result != -8192 )
          break;
        v23 += 48;
        if ( v24 == v23 )
          return result;
      }
      while ( v24 != v23 )
      {
        v25 = *(_QWORD *)(v23 + 24);
        v26 = *(_DWORD *)(v23 + 40);
        v23 += 48;
        v29 = v25;
        v30 = v26;
        result = sub_D45B70((__int64)&v31, v5, &v29);
        if ( v23 == v24 )
          break;
        while ( 1 )
        {
          result = *(_QWORD *)(v23 + 24);
          if ( result != -4096 && result != -8192 )
            break;
          v23 += 48;
          if ( v24 == v23 )
            return result;
        }
      }
    }
  }
  return result;
}
