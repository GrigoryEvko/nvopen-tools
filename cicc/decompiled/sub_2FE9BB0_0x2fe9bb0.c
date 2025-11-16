// Function: sub_2FE9BB0
// Address: 0x2fe9bb0
//
__int64 __fastcall sub_2FE9BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v6; // r14
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int16 v24; // [rsp+Ah] [rbp-86h] BYREF
  unsigned int v25; // [rsp+Ch] [rbp-84h] BYREF
  __int64 v26; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v31; // [rsp+38h] [rbp-58h]
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+48h] [rbp-48h]
  unsigned __int64 v34; // [rsp+50h] [rbp-40h]

  v26 = a4;
  v27 = a5;
  if ( (_WORD)a4 )
    return *(unsigned __int16 *)(a1 + 2LL * (unsigned __int16)a4 + 2852);
  v6 = a2;
  if ( (unsigned __int8)sub_30070B0(&v26, a2, a3) )
  {
    LOWORD(v28) = 0;
    LOWORD(v32) = 0;
    v33 = 0;
    sub_2FE8D10(a1, a2, (unsigned int)v26, a5, &v32, (unsigned int *)&v30, (unsigned __int16 *)&v28);
    return (unsigned __int16)v28;
  }
  if ( !(unsigned __int8)sub_3007070(&v26) )
    goto LABEL_25;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v8 == sub_2D56A50 )
  {
    v9 = a2;
    a2 = a1;
    sub_2FE6CC0((__int64)&v32, a1, v9, v26, v27);
    v10 = v34;
    v11 = (unsigned __int16)v33;
  }
  else
  {
    v11 = v8(a1, a2, v26, a5);
    v10 = v20;
  }
  v12 = (unsigned __int16)v11;
  v28 = v11;
  v29 = v10;
  if ( (_WORD)v11 )
    return *(unsigned __int16 *)(a1 + 2 * v12 + 2852);
  if ( (unsigned __int8)sub_30070B0(&v28, a2, 0) )
  {
    LOWORD(v32) = 0;
    LOWORD(v25) = 0;
    v33 = 0;
    sub_2FE8D10(a1, v6, (unsigned int)v28, v10, &v32, (unsigned int *)&v30, (unsigned __int16 *)&v25);
    return (unsigned __int16)v25;
  }
  if ( !(unsigned __int8)sub_3007070(&v28) )
    goto LABEL_25;
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    v14 = a1;
    sub_2FE6CC0((__int64)&v32, a1, v6, v28, v29);
    v15 = v34;
    v16 = (unsigned __int16)v33;
  }
  else
  {
    v14 = v6;
    v16 = v13(a1, v6, v28, v10);
    v15 = v21;
  }
  v12 = (unsigned __int16)v16;
  v30 = v16;
  v31 = v15;
  if ( (_WORD)v16 )
    return *(unsigned __int16 *)(a1 + 2 * v12 + 2852);
  if ( (unsigned __int8)sub_30070B0(&v30, v14, 0) )
  {
    LOWORD(v32) = 0;
    v24 = 0;
    v33 = 0;
    sub_2FE8D10(a1, v6, (unsigned int)v30, v15, &v32, &v25, &v24);
    return v24;
  }
  if ( !(unsigned __int8)sub_3007070(&v30) )
LABEL_25:
    BUG();
  v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v17 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v32, a1, v6, v30, v31);
    v18 = v34;
    v19 = (unsigned __int16)v33;
  }
  else
  {
    v22 = v17(a1, v6, v30, v15);
    v18 = v23;
    v19 = v22;
  }
  return sub_2FE98B0(a1, v6, v19, v18);
}
