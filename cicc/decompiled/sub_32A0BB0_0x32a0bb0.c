// Function: sub_32A0BB0
// Address: 0x32a0bb0
//
__int64 __fastcall sub_32A0BB0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rdi
  int v7; // esi
  __int64 v8; // r8
  int v9; // r9d
  __int64 *v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // rbx
  __int64 v13; // r12
  int v14; // r14d
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int16 *v22; // rax
  __int64 *v23; // [rsp+8h] [rbp-48h]
  __m128i v24; // [rsp+10h] [rbp-40h]
  __m128i v25; // [rsp+20h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( *((_DWORD *)a4 + 4) != v4 )
  {
LABEL_2:
    if ( (unsigned int)(v4 - 205) <= 1 )
    {
      v5 = *(_QWORD *)(a1 + 40);
      if ( *(_DWORD *)(*(_QWORD *)v5 + 24LL) == 208 )
      {
        v6 = *(_QWORD *)(v5 + 40);
        v7 = *(_DWORD *)(v5 + 48);
        v8 = *(_QWORD *)(v5 + 80);
        v9 = *(_DWORD *)(v5 + 88);
        v10 = *(__int64 **)(*(_QWORD *)v5 + 40LL);
        v11 = *((_DWORD *)v10 + 2);
        v12 = *v10;
        v13 = v10[5];
        v14 = *((_DWORD *)v10 + 12);
        v15 = v10[10];
        if ( v8 == v13 && v6 == *v10 && v7 == v11 && v9 == v14 )
          goto LABEL_6;
        if ( v8 == v12 && v7 == v14 && v6 == v13 && v9 == v11 )
        {
          if ( v6 != v12 || v7 != v11 )
          {
            v23 = a4;
            v22 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * v11);
            v16 = sub_33CBD40(*(unsigned int *)(v15 + 96), *v22, *((_QWORD *)v22 + 1));
            a4 = v23;
LABEL_7:
            if ( (unsigned int)(v16 - 18) <= 1 )
            {
              v17 = *a4;
              *(_QWORD *)v17 = v12;
              *(_DWORD *)(v17 + 8) = v11;
              v18 = a4[1];
              *(_QWORD *)v18 = v13;
              *(_DWORD *)(v18 + 8) = v14;
              return 1;
            }
            return 0;
          }
LABEL_6:
          v16 = *(_DWORD *)(v15 + 96);
          goto LABEL_7;
        }
      }
    }
    return 0;
  }
  v20 = a4[3];
  v25 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  *(_QWORD *)v20 = v25.m128i_i64[0];
  *(_DWORD *)(v20 + 8) = v25.m128i_i32[2];
  v21 = a4[4];
  v24 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
  *(_QWORD *)v21 = v24.m128i_i64[0];
  *(_DWORD *)(v21 + 8) = v24.m128i_i32[2];
  result = *((unsigned __int8 *)a4 + 44);
  if ( !(_BYTE)result )
    return 1;
  if ( *((_DWORD *)a4 + 10) != ((_DWORD)a4[5] & *(_DWORD *)(a1 + 28)) )
  {
    v4 = *(_DWORD *)(a1 + 24);
    goto LABEL_2;
  }
  return result;
}
