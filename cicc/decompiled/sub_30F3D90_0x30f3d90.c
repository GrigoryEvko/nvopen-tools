// Function: sub_30F3D90
// Address: 0x30f3d90
//
void __fastcall sub_30F3D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r9
  signed __int64 v8; // r12
  __int64 v9; // r13
  const __m128i *v10; // r9
  const __m128i *v11; // r10
  const __m128i *v12; // r11
  __int64 v13; // r14
  unsigned __int64 v14; // rax
  __int64 v15; // r10
  int v16; // ecx
  bool v17; // cc
  __int64 v18; // rdx
  __m128i v19; // xmm0
  const __m128i *v21; // [rsp+8h] [rbp-58h]
  unsigned __int64 v22; // [rsp+10h] [rbp-50h]
  const __m128i *v23; // [rsp+18h] [rbp-48h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v15 = a2;
        v14 = a1;
LABEL_12:
        v16 = *(_DWORD *)(v15 + 16);
        v17 = *(_DWORD *)(v14 + 16) < v16;
        if ( *(_DWORD *)(v14 + 16) == v16 )
          v17 = *(_QWORD *)(v14 + 8) < *(_QWORD *)(v15 + 8);
        if ( v17 )
        {
          v18 = *(_QWORD *)v14;
          *(_QWORD *)v14 = *(_QWORD *)v15;
          *(_QWORD *)v15 = v18;
          v19 = _mm_loadu_si128((const __m128i *)(v14 + 8));
          *(_QWORD *)(v14 + 8) = *(_QWORD *)(v15 + 8);
          *(_DWORD *)(v14 + 16) = *(_DWORD *)(v15 + 16);
          *(_QWORD *)(v15 + 8) = v19.m128i_i64[0];
          *(_DWORD *)(v15 + 16) = v19.m128i_i32[2];
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v11 = (const __m128i *)sub_30F3D20(
                                 v7,
                                 a3,
                                 v6 + 8 * (v5 / 2 + ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
        v13 = 0xAAAAAAAAAAAAAAABLL * (((char *)v11 - (char *)v10) >> 3);
        while ( 1 )
        {
          v21 = v11;
          v23 = v12;
          v8 -= v13;
          v22 = sub_30F37F0(v12, v10, v11);
          sub_30F3D90(v6, v23, v22, v9, v13);
          v5 -= v9;
          if ( !v5 )
            break;
          v14 = v22;
          v15 = (__int64)v21;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v7 = (__int64)v21;
          v6 = v22;
          if ( v5 > v8 )
            goto LABEL_5;
LABEL_10:
          v13 = v8 / 2;
          v12 = (const __m128i *)sub_30F3CB0(
                                   v6,
                                   v7,
                                   v7 + 8 * (v8 / 2 + ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
          v9 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v12->m128i_i64 - v6) >> 3);
        }
      }
    }
  }
}
