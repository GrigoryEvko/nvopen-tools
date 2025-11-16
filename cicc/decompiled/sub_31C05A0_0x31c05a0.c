// Function: sub_31C05A0
// Address: 0x31c05a0
//
char __fastcall sub_31C05A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __m128i v13; // xmm1
  _QWORD *v14; // r15
  __int64 v15; // rax
  _BYTE *v16; // r14
  __int64 v17; // rax
  bool v18; // zf
  _BYTE *v19; // rsi
  _BYTE *v20; // rsi
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  _QWORD *v24; // [rsp+20h] [rbp-1F0h]
  __int64 v25; // [rsp+28h] [rbp-1E8h]
  _QWORD v26[5]; // [rsp+38h] [rbp-1D8h] BYREF
  _QWORD v27[12]; // [rsp+60h] [rbp-1B0h] BYREF
  _OWORD v28[2]; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v29; // [rsp+E0h] [rbp-130h]
  __int64 v30; // [rsp+E8h] [rbp-128h]
  __int64 v31; // [rsp+F0h] [rbp-120h]
  __int64 v32; // [rsp+F8h] [rbp-118h]
  __int64 v33; // [rsp+100h] [rbp-110h]
  __int64 v34; // [rsp+108h] [rbp-108h]
  __int64 v35; // [rsp+110h] [rbp-100h]
  __int64 v36; // [rsp+118h] [rbp-F8h]
  _QWORD v37[12]; // [rsp+120h] [rbp-F0h] BYREF
  __m128i v38; // [rsp+180h] [rbp-90h] BYREF
  __m128i v39; // [rsp+190h] [rbp-80h] BYREF
  __int64 v40; // [rsp+1A0h] [rbp-70h]
  __int64 v41; // [rsp+1A8h] [rbp-68h]
  __int64 v42; // [rsp+1B0h] [rbp-60h]
  __int64 v43; // [rsp+1B8h] [rbp-58h]
  __int64 v44; // [rsp+1C0h] [rbp-50h]
  __int64 v45; // [rsp+1C8h] [rbp-48h]
  __int64 v46; // [rsp+1D0h] [rbp-40h]
  __int64 v47; // [rsp+1D8h] [rbp-38h]

  v8 = *(_QWORD *)(a1 + 184);
  v9 = *(_QWORD *)(a1 + 192);
  v10 = *(_QWORD *)(a1 + 176);
  v11 = *(_QWORD *)(a1 + 200);
  v26[2] = v8;
  v26[4] = v11;
  v26[3] = v9;
  v26[1] = v10;
  sub_31C0060((__int64 *)a2, v9, v8, a4, a5, a6, v10, v8);
  v12 = sub_31BFFF0((__int64 **)a2);
  sub_318B480((__int64)&v38, *(_QWORD *)(v12 + 8));
  if ( *(_BYTE *)(a1 + 208) )
  {
    v13 = _mm_loadu_si128(&v39);
    *(__m128i *)(a1 + 176) = _mm_loadu_si128(&v38);
    *(__m128i *)(a1 + 192) = v13;
  }
  else
  {
    v21 = _mm_loadu_si128(&v38);
    v22 = _mm_loadu_si128(&v39);
    *(_BYTE *)(a1 + 208) = 1;
    *(__m128i *)(a1 + 176) = v21;
    *(__m128i *)(a1 + 192) = v22;
  }
  v14 = *(_QWORD **)a2;
  v15 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v24 = (_QWORD *)v15;
  if ( *(_QWORD *)a2 != v15 )
  {
    v25 = a1 + 40;
    do
    {
      v16 = (_BYTE *)*v14;
      (*(void (__fastcall **)(__m128i *, _QWORD, __int64))(*(_QWORD *)*v14 + 24LL))(&v38, *v14, v25);
      (*(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(*(_QWORD *)v16 + 16LL))(v37, v16, v25);
      v27[0] = v37[0];
      v27[1] = v37[1];
      v27[2] = v37[2];
      v27[3] = v37[3];
      v27[4] = v37[4];
      v27[5] = v37[5];
      v27[6] = v37[6];
      v27[7] = v37[7];
      v27[8] = v37[8];
      v27[9] = v37[9];
      v27[10] = v37[10];
      v27[11] = v37[11];
      v28[0] = v38;
      v28[1] = v39;
      v29 = v40;
      v30 = v41;
      v31 = v42;
      v32 = v43;
      v33 = v44;
      v34 = v45;
      v35 = v46;
      v36 = v47;
      while ( 1 )
      {
        LOBYTE(v15) = sub_31B8DE0(v27, v28);
        if ( (_BYTE)v15 )
          break;
        v17 = sub_31B8B80((__int64)v27);
        v18 = (*(_DWORD *)(v17 + 20))-- == 1;
        if ( v18 && !*(_BYTE *)(v17 + 24) )
        {
          v26[0] = v17;
          v19 = *(_BYTE **)(a1 + 16);
          if ( v19 == *(_BYTE **)(a1 + 24) )
          {
            sub_31C0410(a1 + 8, v19, v26);
            v20 = *(_BYTE **)(a1 + 16);
          }
          else
          {
            if ( v19 )
            {
              *(_QWORD *)v19 = v17;
              v19 = *(_BYTE **)(a1 + 16);
            }
            v20 = v19 + 8;
            *(_QWORD *)(a1 + 16) = v20;
          }
          sub_31BFEC0(*(_QWORD *)(a1 + 8), ((__int64)&v20[-*(_QWORD *)(a1 + 8)] >> 3) - 1, 0, *((_QWORD *)v20 - 1));
        }
        sub_31B8D10((__int64)v27);
      }
      v16[24] = 1;
      ++v14;
    }
    while ( v24 != v14 );
  }
  return v15;
}
