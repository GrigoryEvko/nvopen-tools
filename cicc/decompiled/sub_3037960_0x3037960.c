// Function: sub_3037960
// Address: 0x3037960
//
__int64 __fastcall sub_3037960(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, int a5)
{
  __int64 result; // rax
  __int64 v7; // rdx
  int v8; // r14d
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int16 v15; // dx
  __int64 v16; // rax
  char v17; // cl
  __int64 v18; // rax
  unsigned int v19; // eax
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-B8h]
  __int64 v24; // [rsp+10h] [rbp-B0h]
  unsigned __int16 v25; // [rsp+2Ah] [rbp-96h] BYREF
  unsigned int v26; // [rsp+2Ch] [rbp-94h] BYREF
  __int64 v27; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v28; // [rsp+38h] [rbp-88h]
  __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v30; // [rsp+48h] [rbp-78h]
  _QWORD v31[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v32; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v33; // [rsp+68h] [rbp-58h]
  __int64 v34; // [rsp+70h] [rbp-50h] BYREF
  __int64 v35; // [rsp+78h] [rbp-48h]
  unsigned __int64 v36; // [rsp+80h] [rbp-40h]

  if ( (_WORD)a3 != 9 )
  {
    v27 = a3;
    v28 = a4;
    if ( (_WORD)a3 )
      return *(unsigned __int16 *)(a1 + 2LL * (unsigned __int16)a3 + 2304);
    if ( sub_30070B0((__int64)&v27) )
    {
      LOWORD(v34) = 0;
      v35 = 0;
      LOWORD(v31[0]) = 0;
      return sub_2FE8D10(a1, a2, (unsigned int)v27, a4, &v34, (unsigned int *)&v32, (unsigned __int16 *)v31);
    }
    if ( sub_3007070((__int64)&v27) )
    {
      v31[0] = sub_3007260((__int64)&v27);
      v31[1] = v7;
      v34 = v31[0];
      LOBYTE(v35) = v7;
      v8 = sub_CA1930(&v34);
      v29 = v27;
      v30 = v28;
      if ( (_WORD)v27 )
      {
        v15 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v27 + 2852);
        goto LABEL_22;
      }
      v23 = v28;
      v24 = v27;
      if ( sub_30070B0((__int64)&v29) )
      {
        LOWORD(v34) = 0;
        LOWORD(v26) = 0;
        v35 = 0;
        sub_2FE8D10(a1, a2, (unsigned int)v29, v30, &v34, (unsigned int *)&v32, (unsigned __int16 *)&v26);
        v15 = v26;
        goto LABEL_22;
      }
      if ( sub_3007070((__int64)&v29) )
      {
        v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
        if ( v9 == sub_2D56A50 )
        {
          sub_2FE6CC0((__int64)&v34, a1, a2, v24, v23);
          v10 = v36;
          v11 = (unsigned __int16)v35;
        }
        else
        {
          v11 = v9(a1, a2, v29, v30);
          v10 = v20;
        }
        v32 = v11;
        v33 = v10;
        if ( (_WORD)v11 )
        {
          v15 = *(_WORD *)(a1 + 2LL * (unsigned __int16)v11 + 2852);
          goto LABEL_22;
        }
        if ( sub_30070B0((__int64)&v32) )
        {
          LOWORD(v34) = 0;
          v25 = 0;
          v35 = 0;
          sub_2FE8D10(a1, a2, (unsigned int)v32, v10, &v34, &v26, &v25);
          v15 = v25;
          goto LABEL_22;
        }
        if ( sub_3007070((__int64)&v32) )
        {
          v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
          if ( v12 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v34, a1, a2, v32, v33);
            v13 = v36;
            v14 = (unsigned __int16)v35;
          }
          else
          {
            v21 = v12(a1, a2, v32, v10);
            v13 = v22;
            v14 = v21;
          }
          v15 = sub_2FE98B0(a1, a2, v14, v13);
LABEL_22:
          if ( v15 <= 1u || (unsigned __int16)(v15 - 504) <= 7u )
            BUG();
          v16 = 16LL * (v15 - 1);
          v17 = byte_444C4A0[v16 + 8];
          v18 = *(_QWORD *)&byte_444C4A0[v16];
          LOBYTE(v35) = v17;
          v34 = v18;
          v19 = sub_CA1930(&v34);
          return (v8 + v19 - 1) / v19;
        }
      }
    }
    BUG();
  }
  if ( !BYTE2(a5) )
    return *(unsigned __int16 *)(a1 + 2LL * (unsigned __int16)a3 + 2304);
  result = 1;
  if ( (_WORD)a5 != 9 )
    return *(unsigned __int16 *)(a1 + 2LL * (unsigned __int16)a3 + 2304);
  return result;
}
