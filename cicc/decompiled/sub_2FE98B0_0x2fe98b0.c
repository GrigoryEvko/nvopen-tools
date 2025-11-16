// Function: sub_2FE98B0
// Address: 0x2fe98b0
//
__int64 __fastcall sub_2FE98B0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v5; // r14
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-90h] BYREF
  unsigned __int64 v24; // [rsp+8h] [rbp-88h]
  unsigned __int16 v25; // [rsp+1Ah] [rbp-76h] BYREF
  unsigned int v26; // [rsp+1Ch] [rbp-74h] BYREF
  __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v28; // [rsp+28h] [rbp-68h]
  __int64 v29; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v30; // [rsp+38h] [rbp-58h]
  __int64 v31; // [rsp+40h] [rbp-50h] BYREF
  __int64 v32; // [rsp+48h] [rbp-48h]
  unsigned __int64 v33; // [rsp+50h] [rbp-40h]

  v23 = a3;
  v24 = a4;
  if ( (_WORD)a3 )
    return *(unsigned __int16 *)(a1 + 2LL * (unsigned __int16)a3 + 2852);
  v5 = a2;
  if ( (unsigned __int8)sub_30070B0(&v23, a2, a3) )
  {
    LOWORD(v27) = 0;
    LOWORD(v31) = 0;
    v32 = 0;
    sub_2FE8D10(a1, a2, (unsigned int)v23, a4, &v31, (unsigned int *)&v29, (unsigned __int16 *)&v27);
    return (unsigned __int16)v27;
  }
  if ( !(unsigned __int8)sub_3007070(&v23) )
    goto LABEL_25;
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    v8 = a2;
    a2 = a1;
    sub_2FE6CC0((__int64)&v31, a1, v8, v23, v24);
    v9 = v33;
    v10 = (unsigned __int16)v32;
  }
  else
  {
    v10 = v7(a1, a2, v23, a4);
    v9 = v19;
  }
  v11 = (unsigned __int16)v10;
  v27 = v10;
  v28 = v9;
  if ( (_WORD)v10 )
    return *(unsigned __int16 *)(a1 + 2 * v11 + 2852);
  if ( (unsigned __int8)sub_30070B0(&v27, a2, 0) )
  {
    LOWORD(v31) = 0;
    LOWORD(v26) = 0;
    v32 = 0;
    sub_2FE8D10(a1, v5, (unsigned int)v27, v9, &v31, (unsigned int *)&v29, (unsigned __int16 *)&v26);
    return (unsigned __int16)v26;
  }
  if ( !(unsigned __int8)sub_3007070(&v27) )
    goto LABEL_25;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    v13 = a1;
    sub_2FE6CC0((__int64)&v31, a1, v5, v27, v28);
    v14 = v33;
    v15 = (unsigned __int16)v32;
  }
  else
  {
    v13 = v5;
    v15 = v12(a1, v5, v27, v9);
    v14 = v20;
  }
  v11 = (unsigned __int16)v15;
  v29 = v15;
  v30 = v14;
  if ( (_WORD)v15 )
    return *(unsigned __int16 *)(a1 + 2 * v11 + 2852);
  if ( (unsigned __int8)sub_30070B0(&v29, v13, 0) )
  {
    LOWORD(v31) = 0;
    v25 = 0;
    v32 = 0;
    sub_2FE8D10(a1, v5, (unsigned int)v29, v14, &v31, &v26, &v25);
    return v25;
  }
  if ( !(unsigned __int8)sub_3007070(&v29) )
LABEL_25:
    BUG();
  v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v16 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v31, a1, v5, v29, v30);
    v17 = v33;
    v18 = (unsigned __int16)v32;
  }
  else
  {
    v21 = v16(a1, v5, v29, v14);
    v17 = v22;
    v18 = v21;
  }
  return sub_2FE98B0(a1, v5, v18, v17);
}
