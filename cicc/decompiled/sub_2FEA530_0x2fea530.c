// Function: sub_2FEA530
// Address: 0x2fea530
//
__int64 __fastcall sub_2FEA530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // r14
  __int64 v7; // rax
  __int64 (__fastcall *v8)(__int64, __int64, __int64, unsigned __int64); // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  int v11; // r13d
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int16 v19; // dx
  __int64 v21; // rax
  char v22; // cl
  __int64 v23; // rax
  unsigned int v24; // eax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // [rsp+0h] [rbp-B0h]
  __int64 v29; // [rsp+8h] [rbp-A8h]
  unsigned __int16 v30; // [rsp+16h] [rbp-9Ah] BYREF
  unsigned int v31; // [rsp+18h] [rbp-98h] BYREF
  unsigned int v32; // [rsp+1Ch] [rbp-94h]
  __int64 v33; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v34; // [rsp+28h] [rbp-88h]
  __int64 v35; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v36; // [rsp+38h] [rbp-78h]
  _QWORD v37[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v38; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v39; // [rsp+58h] [rbp-58h]
  __int64 v40; // [rsp+60h] [rbp-50h] BYREF
  __int64 v41; // [rsp+68h] [rbp-48h]
  unsigned __int64 v42; // [rsp+70h] [rbp-40h]

  v5 = a2;
  v7 = *(_QWORD *)a1;
  BYTE2(v32) = 0;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(v7 + 736);
  if ( v8 != sub_2FEA1A0 )
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD, unsigned __int64, _QWORD))v8)(
             a1,
             a2,
             (unsigned int)a4,
             a5,
             v32);
  v33 = a4;
  v34 = a5;
  if ( (_WORD)a4 )
    return *(unsigned __int16 *)(a1 + 2LL * (unsigned __int16)a4 + 2304);
  if ( !(unsigned __int8)sub_30070B0(&v33, a2, a3) )
  {
    if ( !(unsigned __int8)sub_3007070(&v33) )
      goto LABEL_28;
    v37[0] = sub_3007260(&v33);
    v37[1] = v9;
    v40 = v37[0];
    LOBYTE(v41) = v9;
    v11 = sub_CA1930(&v40);
    v35 = v33;
    v36 = v34;
    if ( (_WORD)v33 )
    {
      v19 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v33 + 2852);
    }
    else
    {
      v28 = v34;
      v29 = v33;
      if ( (unsigned __int8)sub_30070B0(&v35, a2, v10) )
      {
        LOWORD(v40) = 0;
        LOWORD(v31) = 0;
        v41 = 0;
        sub_2FE8D10(a1, a2, (unsigned int)v35, v36, &v40, (unsigned int *)&v38, (unsigned __int16 *)&v31);
        v19 = v31;
      }
      else
      {
        if ( !(unsigned __int8)sub_3007070(&v35) )
          goto LABEL_28;
        v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
        if ( v12 == sub_2D56A50 )
        {
          v13 = a2;
          a2 = a1;
          sub_2FE6CC0((__int64)&v40, a1, v13, v29, v28);
          v14 = v42;
          v15 = (unsigned __int16)v41;
        }
        else
        {
          v15 = v12(a1, a2, v35, v36);
          v14 = v25;
        }
        v38 = v15;
        v39 = v14;
        if ( (_WORD)v15 )
        {
          v19 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v15 + 2852);
        }
        else if ( (unsigned __int8)sub_30070B0(&v38, a2, 0) )
        {
          LOWORD(v40) = 0;
          v30 = 0;
          v41 = 0;
          sub_2FE8D10(a1, v5, (unsigned int)v38, v14, &v40, &v31, &v30);
          v19 = v30;
        }
        else
        {
          if ( !(unsigned __int8)sub_3007070(&v38) )
            goto LABEL_28;
          v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
          if ( v16 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v40, a1, v5, v38, v39);
            v17 = v42;
            v18 = (unsigned __int16)v41;
          }
          else
          {
            v26 = v16(a1, v5, v38, v14);
            v17 = v27;
            v18 = v26;
          }
          v19 = sub_2FE98B0(a1, v5, v18, v17);
        }
      }
    }
    if ( v19 > 1u && (unsigned __int16)(v19 - 504) > 7u )
    {
      v21 = 16LL * (v19 - 1);
      v22 = byte_444C4A0[v21 + 8];
      v23 = *(_QWORD *)&byte_444C4A0[v21];
      LOBYTE(v41) = v22;
      v40 = v23;
      v24 = sub_CA1930(&v40);
      return (v11 + v24 - 1) / v24;
    }
LABEL_28:
    BUG();
  }
  LOWORD(v37[0]) = 0;
  LOWORD(v40) = 0;
  v41 = 0;
  return sub_2FE8D10(a1, a2, (unsigned int)v33, a5, &v40, (unsigned int *)&v38, (unsigned __int16 *)v37);
}
