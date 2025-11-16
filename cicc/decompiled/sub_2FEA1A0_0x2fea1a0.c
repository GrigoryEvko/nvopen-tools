// Function: sub_2FEA1A0
// Address: 0x2fea1a0
//
__int64 __fastcall sub_2FEA1A0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v5; // r14
  __int64 v7; // rdx
  __int64 v8; // rdx
  int v9; // r12d
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  __int64 (__fastcall *v14)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned __int16 v17; // dx
  __int64 v18; // rax
  char v19; // cl
  __int64 v20; // rax
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-B8h]
  __int64 v26; // [rsp+10h] [rbp-B0h]
  __int64 v27; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v28; // [rsp+28h] [rbp-98h]
  unsigned __int16 v29; // [rsp+3Ah] [rbp-86h] BYREF
  unsigned int v30; // [rsp+3Ch] [rbp-84h] BYREF
  __int64 v31; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v32; // [rsp+48h] [rbp-78h]
  _QWORD v33[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v34; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v35; // [rsp+68h] [rbp-58h]
  __int64 v36; // [rsp+70h] [rbp-50h] BYREF
  __int64 v37; // [rsp+78h] [rbp-48h]
  unsigned __int64 v38; // [rsp+80h] [rbp-40h]

  v27 = a3;
  v28 = a4;
  if ( (_WORD)a3 )
    return *(unsigned __int16 *)(a1 + 2LL * (unsigned __int16)a3 + 2304);
  v5 = a2;
  if ( (unsigned __int8)sub_30070B0(&v27, a2, a3) )
  {
    LOWORD(v33[0]) = 0;
    LOWORD(v36) = 0;
    v37 = 0;
    return sub_2FE8D10(a1, a2, (unsigned int)v27, a4, &v36, (unsigned int *)&v34, (unsigned __int16 *)v33);
  }
  if ( !(unsigned __int8)sub_3007070(&v27) )
    goto LABEL_26;
  v33[0] = sub_3007260(&v27);
  v33[1] = v7;
  v36 = v33[0];
  LOBYTE(v37) = v7;
  v9 = sub_CA1930(&v36);
  v31 = v27;
  v32 = v28;
  if ( (_WORD)v27 )
  {
    v17 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v27 + 2852);
  }
  else
  {
    v25 = v28;
    v26 = v27;
    if ( (unsigned __int8)sub_30070B0(&v31, a2, v8) )
    {
      LOWORD(v36) = 0;
      LOWORD(v30) = 0;
      v37 = 0;
      sub_2FE8D10(a1, a2, (unsigned int)v31, v32, &v36, (unsigned int *)&v34, (unsigned __int16 *)&v30);
      v17 = v30;
    }
    else
    {
      if ( !(unsigned __int8)sub_3007070(&v31) )
        goto LABEL_26;
      v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
      if ( v10 == sub_2D56A50 )
      {
        v11 = a2;
        a2 = a1;
        sub_2FE6CC0((__int64)&v36, a1, v11, v26, v25);
        v12 = v38;
        v13 = (unsigned __int16)v37;
      }
      else
      {
        v13 = v10(a1, a2, v31, v32);
        v12 = v22;
      }
      v34 = v13;
      v35 = v12;
      if ( (_WORD)v13 )
      {
        v17 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v13 + 2852);
      }
      else if ( (unsigned __int8)sub_30070B0(&v34, a2, 0) )
      {
        LOWORD(v36) = 0;
        v29 = 0;
        v37 = 0;
        sub_2FE8D10(a1, v5, (unsigned int)v34, v12, &v36, &v30, &v29);
        v17 = v29;
      }
      else
      {
        if ( !(unsigned __int8)sub_3007070(&v34) )
          goto LABEL_26;
        v14 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
        if ( v14 == sub_2D56A50 )
        {
          sub_2FE6CC0((__int64)&v36, a1, v5, v34, v35);
          v15 = v38;
          v16 = (unsigned __int16)v37;
        }
        else
        {
          v23 = v14(a1, v5, v34, v12);
          v15 = v24;
          v16 = v23;
        }
        v17 = sub_2FE98B0(a1, v5, v16, v15);
      }
    }
  }
  if ( v17 <= 1u || (unsigned __int16)(v17 - 504) <= 7u )
LABEL_26:
    BUG();
  v18 = 16LL * (v17 - 1);
  v19 = byte_444C4A0[v18 + 8];
  v20 = *(_QWORD *)&byte_444C4A0[v18];
  LOBYTE(v37) = v19;
  v36 = v20;
  v21 = sub_CA1930(&v36);
  return (v9 + v21 - 1) / v21;
}
