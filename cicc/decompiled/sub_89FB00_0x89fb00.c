// Function: sub_89FB00
// Address: 0x89fb00
//
__int64 __fastcall sub_89FB00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6)
{
  unsigned int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rbx
  __m128i *v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r13
  __int8 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  _DWORD *v20; // r12
  __int64 v23; // [rsp+8h] [rbp-E8h]
  __int64 v24; // [rsp+8h] [rbp-E8h]
  __int64 v26; // [rsp+10h] [rbp-E0h]
  __int64 v27; // [rsp+10h] [rbp-E0h]
  _QWORD v29[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v30[2]; // [rsp+30h] [rbp-C0h] BYREF
  const __m128i *v31; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v32; // [rsp+48h] [rbp-A8h]
  __int64 v33; // [rsp+50h] [rbp-A0h]
  _QWORD v34[18]; // [rsp+60h] [rbp-90h] BYREF

  v6 = 1;
  v7 = *(_QWORD *)(a1 + 168);
  if ( *(_DWORD *)(v7 + 28) != -2 )
  {
    v8 = *(_QWORD *)(v7 + 32);
    if ( v8 )
    {
      v10 = 0;
      v33 = 0;
      v32 = 1;
      v31 = (const __m128i *)sub_823970(24);
      v11 = a4;
      v12 = a3;
      if ( a6 )
      {
        v29[0] = 0;
        v10 = (__m128i *)v29;
        v29[1] = 0;
      }
      if ( a5 )
      {
        v23 = a4;
        v26 = v12;
        sub_89F970(a5, (__int64)&v31);
        v13 = v32;
        v12 = v26;
        v11 = v23;
      }
      else
      {
        v13 = 1;
      }
      v14 = v33;
      if ( v13 == v33 )
      {
        v24 = v11;
        v27 = v12;
        sub_738390(&v31);
        v11 = v24;
        v12 = v27;
      }
      v15 = &v31->m128i_i8[24 * v14];
      if ( v15 )
      {
        v15[16] &= 0xF0u;
        *(_QWORD *)v15 = v11;
        *((_QWORD *)v15 + 1) = v12;
      }
      v33 = v14 + 1;
      sub_892150(v34);
      v6 = sub_6F1D40(a2, v8, (int)&v31, (__int64)v34, v10);
      if ( v6 )
      {
        v6 = 1;
      }
      else if ( a6 )
      {
        v30[0] = 0;
        v30[1] = 0;
        sub_686CA0(0xBF5u, v8 + 28, a2, v30);
        sub_67E390(v30, v10->m128i_i64, 0);
        v20 = sub_67D9D0(0xBE4u, a6);
        sub_67E370((__int64)v20, v10);
        sub_685910((__int64)v20, (FILE *)v10);
      }
      sub_823A00((__int64)v31, 24 * v32, v16, v17, v18, v19);
    }
  }
  return v6;
}
