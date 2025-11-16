// Function: sub_34D06B0
// Address: 0x34d06b0
//
__int64 __fastcall sub_34D06B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // r14
  unsigned __int64 v5; // r13
  __int64 v6; // r12
  __int64 (__fastcall *v7)(__int64, __int64, __int64, unsigned __int64); // r9
  __int64 v8; // rdx
  int v9; // ebx
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int16 v16; // dx
  __int64 v18; // rax
  char v19; // cl
  __int64 v20; // rax
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // [rsp+0h] [rbp-B0h]
  __int64 v26; // [rsp+8h] [rbp-A8h]
  unsigned __int16 v27; // [rsp+16h] [rbp-9Ah] BYREF
  unsigned int v28; // [rsp+18h] [rbp-98h] BYREF
  unsigned int v29; // [rsp+1Ch] [rbp-94h]
  __int64 v30; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v31; // [rsp+28h] [rbp-88h]
  __int64 v32; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v33; // [rsp+38h] [rbp-78h]
  _QWORD v34[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v35; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v36; // [rsp+58h] [rbp-58h]
  __int64 v37; // [rsp+60h] [rbp-50h] BYREF
  __int64 v38; // [rsp+68h] [rbp-48h]
  unsigned __int64 v39; // [rsp+70h] [rbp-40h]

  v2 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), a2, 0);
  v4 = *(_QWORD *)(a1 + 24);
  BYTE2(v29) = 0;
  v5 = v3;
  v6 = *a2;
  v7 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v4 + 736LL);
  if ( v7 != sub_2FEA1A0 )
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD, unsigned __int64, _QWORD))v7)(
             v4,
             v6,
             (unsigned int)v2,
             v3,
             v29);
  v30 = v2;
  v31 = v3;
  if ( (_WORD)v2 )
    return *(unsigned __int16 *)(v4 + 2LL * (unsigned __int16)v2 + 2304);
  if ( !sub_30070B0((__int64)&v30) )
  {
    if ( !sub_3007070((__int64)&v30) )
      goto LABEL_28;
    v34[0] = sub_3007260((__int64)&v30);
    v34[1] = v8;
    v37 = v34[0];
    LOBYTE(v38) = v8;
    v9 = sub_CA1930(&v37);
    v32 = v30;
    v33 = v31;
    if ( (_WORD)v30 )
    {
      v16 = *(_WORD *)(v4 + 2LL * (unsigned __int16)v30 + 2852);
    }
    else
    {
      v25 = v31;
      v26 = v30;
      if ( sub_30070B0((__int64)&v32) )
      {
        LOWORD(v37) = 0;
        LOWORD(v28) = 0;
        v38 = 0;
        sub_2FE8D10(v4, v6, (unsigned int)v32, v33, &v37, (unsigned int *)&v35, (unsigned __int16 *)&v28);
        v16 = v28;
      }
      else
      {
        if ( !sub_3007070((__int64)&v32) )
          goto LABEL_28;
        v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
        if ( v10 == sub_2D56A50 )
        {
          sub_2FE6CC0((__int64)&v37, v4, v6, v26, v25);
          v11 = v39;
          v12 = (unsigned __int16)v38;
        }
        else
        {
          v12 = v10(v4, v6, v32, v33);
          v11 = v22;
        }
        v35 = v12;
        v36 = v11;
        if ( (_WORD)v12 )
        {
          v16 = *(_WORD *)(v4 + 2LL * (unsigned __int16)v12 + 2852);
        }
        else if ( sub_30070B0((__int64)&v35) )
        {
          LOWORD(v37) = 0;
          v27 = 0;
          v38 = 0;
          sub_2FE8D10(v4, v6, (unsigned int)v35, v11, &v37, &v28, &v27);
          v16 = v27;
        }
        else
        {
          if ( !sub_3007070((__int64)&v35) )
            goto LABEL_28;
          v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
          if ( v13 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v37, v4, v6, v35, v36);
            v14 = v39;
            v15 = (unsigned __int16)v38;
          }
          else
          {
            v23 = v13(v4, v6, v35, v11);
            v14 = v24;
            v15 = v23;
          }
          v16 = sub_2FE98B0(v4, v6, v15, v14);
        }
      }
    }
    if ( v16 > 1u && (unsigned __int16)(v16 - 504) > 7u )
    {
      v18 = 16LL * (v16 - 1);
      v19 = byte_444C4A0[v18 + 8];
      v20 = *(_QWORD *)&byte_444C4A0[v18];
      LOBYTE(v38) = v19;
      v37 = v20;
      v21 = sub_CA1930(&v37);
      return (v9 + v21 - 1) / v21;
    }
LABEL_28:
    BUG();
  }
  LOWORD(v34[0]) = 0;
  LOWORD(v37) = 0;
  v38 = 0;
  return sub_2FE8D10(v4, v6, (unsigned int)v30, v5, &v37, (unsigned int *)&v35, (unsigned __int16 *)v34);
}
