// Function: sub_1BB3F50
// Address: 0x1bb3f50
//
__int64 __fastcall sub_1BB3F50(__int64 a1, unsigned __int8 a2, unsigned int a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned int v10; // r12d
  unsigned int v11; // eax
  __int64 v12; // rcx
  unsigned int v13; // r14d
  int v15; // r8d
  int v16; // r9d
  unsigned int v17; // r13d
  unsigned __int64 v18; // rcx
  unsigned int v19; // eax
  _BYTE *v20; // rdx
  unsigned int v21; // edx
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // [rsp+4h] [rbp-CCh]
  __int64 v25; // [rsp+10h] [rbp-C0h]
  unsigned int v27; // [rsp+1Ch] [rbp-B4h]
  _BYTE *v28; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+28h] [rbp-A8h]
  _BYTE v30[32]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v31; // [rsp+50h] [rbp-80h] BYREF
  __int64 v32; // [rsp+58h] [rbp-78h]
  __int64 v33; // [rsp+60h] [rbp-70h] BYREF
  int v34; // [rsp+68h] [rbp-68h]
  __int64 v35; // [rsp+70h] [rbp-60h]
  __int64 v36; // [rsp+78h] [rbp-58h]
  __int64 v37; // [rsp+80h] [rbp-50h]

  sub_14C5F20(
    (__int64)&v31,
    *(_QWORD *)(*(_QWORD *)(a1 + 296) + 32LL),
    (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 296) + 40LL) - *(_QWORD *)(*(_QWORD *)(a1 + 296) + 32LL)) >> 3,
    *(_QWORD *)(a1 + 344),
    *(_QWORD *)(a1 + 328));
  j___libc_free_0(*(_QWORD *)(a1 + 16));
  v7 = v32;
  v8 = *(_QWORD *)(a1 + 40);
  v32 = 0;
  v9 = *(_QWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = v7;
  v31 = (__int64 *)((char *)v31 + 1);
  *(_QWORD *)(a1 + 24) = v33;
  v33 = 0;
  *(_DWORD *)(a1 + 32) = v34;
  v34 = 0;
  *(_QWORD *)(a1 + 40) = v35;
  v35 = 0;
  *(_QWORD *)(a1 + 48) = v36;
  v36 = 0;
  *(_QWORD *)(a1 + 56) = v37;
  v37 = 0;
  if ( v8 )
  {
    j_j___libc_free_0(v8, v9 - v8);
    if ( v35 )
      j_j___libc_free_0(v35, v37 - v35);
  }
  v10 = 1;
  j___libc_free_0(v32);
  v25 = sub_1BA5530(a1);
  v11 = sub_14A3170(*(__int64 **)(a1 + 328), 1u);
  v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 320) + 48LL) + 16LL) + 208LL);
  if ( v11 < (unsigned int)v12 )
    LODWORD(v12) = v11;
  if ( HIDWORD(v25) <= (unsigned int)v12 )
  {
    v13 = (unsigned int)v12 / HIDWORD(v25);
    v10 = (unsigned int)v12 / HIDWORD(v25);
    if ( !a3 || (unsigned int)v12 / HIDWORD(v25) <= a3 || (a3 & (a3 - 1)) != 0 )
    {
      v24 = v12;
      if ( (unsigned __int8)sub_14A31D0(*(__int64 **)(a1 + 328), a2) || byte_4FB8D60 == 1 && !a2 )
      {
        v29 = 0x800000000LL;
        v17 = 2 * v13;
        v18 = 0;
        v19 = v24 / (unsigned int)v25;
        v28 = v30;
        v20 = v30;
        if ( 2 * v13 <= v24 / (unsigned int)v25 )
        {
          while ( 1 )
          {
            *(_DWORD *)&v20[4 * v18] = v17;
            v17 *= 2;
            v18 = (unsigned int)(v29 + 1);
            LODWORD(v29) = v29 + 1;
            if ( v17 > v19 )
              break;
            if ( HIDWORD(v29) <= (unsigned int)v18 )
            {
              v27 = v19;
              sub_16CD150((__int64)&v28, v30, 0, 4, v15, v16);
              v18 = (unsigned int)v29;
              v19 = v27;
            }
            v20 = v28;
          }
          v20 = v28;
        }
        sub_1BB2200(&v31, (_QWORD *)a1, (__int64)v20, v18, a4, a5);
        v21 = sub_14A3140(*(__int64 **)(a1 + 328), 1u);
        LODWORD(v22) = v32 - 1;
        if ( (int)v32 - 1 >= 0 )
        {
          v22 = (int)v22;
          while ( HIDWORD(v31[v22]) > v21 )
          {
            if ( (_DWORD)--v22 == -1 )
              goto LABEL_27;
          }
          v10 = *(_DWORD *)&v28[4 * v22];
        }
LABEL_27:
        v23 = sub_14A3200(*(_QWORD *)(a1 + 328));
        if ( v23 && v10 < v23 )
          v10 = v23;
        if ( v31 != &v33 )
          _libc_free((unsigned __int64)v31);
        if ( v28 != v30 )
          _libc_free((unsigned __int64)v28);
      }
    }
    else
    {
      return a3;
    }
  }
  return v10;
}
