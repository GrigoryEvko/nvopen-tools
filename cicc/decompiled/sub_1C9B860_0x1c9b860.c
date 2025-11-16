// Function: sub_1C9B860
// Address: 0x1c9b860
//
void __fastcall sub_1C9B860(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v14; // r13
  double v15; // xmm4_8
  double v16; // xmm5_8
  unsigned __int64 *v17; // rdx
  __int64 v18; // rdi
  unsigned __int64 *v19; // r9
  unsigned __int64 *v20; // r8
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rcx
  unsigned __int64 *v24; // rsi
  _BYTE *v25; // rsi
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 *v32; // rax
  __int64 v33; // r9
  unsigned __int64 v34; // rdx
  __int64 v35; // rdx
  unsigned __int64 *v36; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int64 *v37[5]; // [rsp+18h] [rbp-28h] BYREF

  v36 = (unsigned __int64 *)a2;
  v11 = *(_QWORD *)a2;
  if ( *(_BYTE *)(v11 + 8) == 13 )
  {
    v12 = (__int64)(a1 + 53);
    v14 = sub_1C9A5E0(*(_QWORD *)(a2 - 24), v11, a2, *(_BYTE *)(a2 + 18) & 1, (__int64)(a1 + 53));
    if ( v14 )
    {
      v17 = (unsigned __int64 *)a1[49];
      v18 = (__int64)v36;
      v19 = a1 + 48;
      if ( v17 )
      {
        v20 = a1 + 48;
        v21 = (unsigned __int64 *)a1[49];
        do
        {
          while ( 1 )
          {
            v22 = v21[2];
            v23 = v21[3];
            if ( v21[4] >= (unsigned __int64)v36 )
              break;
            v21 = (unsigned __int64 *)v21[3];
            if ( !v23 )
              goto LABEL_8;
          }
          v20 = v21;
          v21 = (unsigned __int64 *)v21[2];
        }
        while ( v22 );
LABEL_8:
        if ( v19 != v20 )
        {
          v24 = a1 + 48;
          if ( v20[4] <= (unsigned __int64)v36 )
          {
            do
            {
              while ( 1 )
              {
                v26 = v17[2];
                v27 = v17[3];
                if ( v17[4] >= (unsigned __int64)v36 )
                  break;
                v17 = (unsigned __int64 *)v17[3];
                if ( !v27 )
                  goto LABEL_18;
              }
              v24 = v17;
              v17 = (unsigned __int64 *)v17[2];
            }
            while ( v26 );
LABEL_18:
            if ( v19 == v24 || v24[4] > (unsigned __int64)v36 )
            {
              v37[0] = (unsigned __int64 *)&v36;
              v24 = sub_1C9B790(a1 + 47, v24, v37);
            }
            v28 = v24[5];
            v29 = (__int64)(v24[6] - v28) >> 3;
            if ( (_DWORD)v29 )
            {
              v30 = 0;
              v31 = 8LL * (unsigned int)(v29 - 1);
              while ( 1 )
              {
                v32 = (__int64 *)(*(_QWORD *)(v28 + v30)
                                - 24LL * (*(_DWORD *)(*(_QWORD *)(v28 + v30) + 20LL) & 0xFFFFFFF));
                if ( *v32 )
                {
                  v33 = v32[1];
                  v34 = v32[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v34 = v33;
                  if ( v33 )
                    *(_QWORD *)(v33 + 16) = *(_QWORD *)(v33 + 16) & 3LL | v34;
                }
                *v32 = v14;
                v35 = *(_QWORD *)(v14 + 8);
                v32[1] = v35;
                if ( v35 )
                  *(_QWORD *)(v35 + 16) = (unsigned __int64)(v32 + 1) | *(_QWORD *)(v35 + 16) & 3LL;
                v32[2] = (v14 + 8) | v32[2] & 3;
                *(_QWORD *)(v14 + 8) = v32;
                if ( v31 == v30 )
                  break;
                v28 = v24[5];
                v30 += 8;
              }
            }
            v18 = (__int64)v36;
          }
        }
      }
      sub_164D160(v18, v14, a3, a4, a5, a6, v15, v16, a9, a10);
      v25 = (_BYTE *)a1[54];
      v37[0] = v36;
      if ( v25 == (_BYTE *)a1[55] )
      {
        sub_17C2330(v12, v25, v37);
      }
      else
      {
        if ( v25 )
        {
          *(_QWORD *)v25 = v36;
          v25 = (_BYTE *)a1[54];
        }
        a1[54] = v25 + 8;
      }
    }
  }
}
