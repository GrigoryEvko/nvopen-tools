// Function: sub_2AADC00
// Address: 0x2aadc00
//
__int64 __fastcall sub_2AADC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 result; // rax
  bool v8; // cc
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // r8
  __int64 v15; // rcx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r8
  __int64 v21; // rdi
  _QWORD *v22; // r14
  unsigned __int64 v23; // rax
  __int64 v24; // r14
  _QWORD *v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // rdi
  _QWORD *v28; // r14
  unsigned __int64 v29; // rax
  __int64 v30; // r14
  _QWORD *v31; // rdi
  _QWORD *v32; // rax
  __int64 v33; // rsi
  unsigned __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // rcx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rax
  __int64 v40; // [rsp-10h] [rbp-D0h]
  __int64 v41; // [rsp+8h] [rbp-B8h]
  __int64 v42; // [rsp+10h] [rbp-B0h]
  const char *v43; // [rsp+18h] [rbp-A8h]
  __int64 v44; // [rsp+18h] [rbp-A8h]
  __int64 v45; // [rsp+20h] [rbp-A0h]
  __int64 v46; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+28h] [rbp-98h]
  __int64 v50; // [rsp+48h] [rbp-78h] BYREF
  _QWORD v51[2]; // [rsp+50h] [rbp-70h] BYREF
  const char *v52; // [rsp+60h] [rbp-60h] BYREF
  __int64 v53; // [rsp+68h] [rbp-58h]
  char v54; // [rsp+80h] [rbp-40h]
  char v55; // [rsp+81h] [rbp-3Fh]

  result = *(_QWORD *)(a3 + 8);
  v8 = *(_DWORD *)(result + 304) <= (unsigned int)qword_500EA08;
  *(_BYTE *)(a1 + 1800) = *(_DWORD *)(result + 304) > (unsigned int)qword_500EA08;
  if ( !v8 )
    return result;
  v47 = **(_QWORD **)(a2 + 32);
  v12 = sub_D4B130(a2);
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a4 + 8LL))(a4);
  if ( !(_BYTE)result )
  {
    v35 = *(_QWORD *)(a1 + 40);
    v36 = *(_QWORD *)(a1 + 32);
    v55 = 1;
    v54 = 3;
    v44 = v35;
    v46 = v36;
    v52 = "vector.scevcheck";
    v37 = sub_986580(v12);
    v38 = sub_F36960(v12, (__int64 *)(v37 + 24), 0, v46, v44, 0, (void **)&v52, 0);
    *(_QWORD *)a1 = v38;
    v39 = sub_986580(v38);
    result = (__int64)sub_F8C220(a1 + 56, a4, v39);
    *(_QWORD *)(a1 + 8) = result;
  }
  v45 = *(_QWORD *)(a3 + 8);
  if ( *(_BYTE *)v45 )
  {
    v13 = *(_QWORD *)a1;
    v14 = *(_QWORD *)(a1 + 40);
    v55 = 1;
    v15 = *(_QWORD *)(a1 + 32);
    v52 = "vector.memcheck";
    v41 = v14;
    if ( !v13 )
      v13 = v12;
    v42 = v15;
    v54 = 3;
    v16 = sub_986580(v13);
    v17 = sub_F36960(v13, (__int64 *)(v16 + 24), 0, v42, v41, 0, (void **)&v52, 0);
    *(_QWORD *)(a1 + 16) = v17;
    if ( *(_BYTE *)(v45 + 376) )
    {
      v52 = *(const char **)(v45 + 384);
      v43 = v52;
      v53 = *(unsigned int *)(v45 + 392);
      v50 = 0;
      v51[0] = a5;
      v51[1] = &v50;
      v34 = sub_986580(v17);
      *(_QWORD *)(a1 + 24) = sub_F75950(
                               v34,
                               (__int64)v43,
                               v53,
                               (__int64 **)(a1 + 928),
                               (__int64 (__fastcall *)(__int64, _BYTE **, _QWORD))sub_2AB2720,
                               (__int64)v51,
                               a6);
      result = v40;
    }
    else
    {
      v18 = sub_986580(v17);
      result = sub_F73BC0(v18, (__int64 *)a2, v45 + 296, (__int64 *)(a1 + 928), unk_4F86D30);
      *(_QWORD *)(a1 + 24) = result;
    }
  }
  v19 = *(_QWORD *)(a1 + 16);
  v20 = *(_QWORD *)a1;
  if ( v19 )
  {
    if ( !v20 )
      goto LABEL_12;
  }
  else if ( !v20 )
  {
    return result;
  }
  sub_BD84D0(*(_QWORD *)a1, v12);
  v19 = *(_QWORD *)(a1 + 16);
  if ( v19 )
  {
LABEL_12:
    sub_BD84D0(v19, v12);
    v21 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 )
    {
LABEL_16:
      v27 = *(_QWORD *)(a1 + 16);
      if ( v27 )
      {
        v28 = (_QWORD *)sub_986580(v27);
        v29 = sub_986580(v12);
        sub_B444E0(v28, v29 + 24, 0);
        v30 = sub_AA48A0(v12);
        sub_B43C20((__int64)&v52, *(_QWORD *)(a1 + 16));
        v31 = sub_BD2C40(72, unk_3F148B8);
        if ( v31 )
          sub_B4C8A0((__int64)v31, v30, (__int64)v52, v53);
        v32 = (_QWORD *)sub_986580(v12);
        sub_B43D60(v32);
      }
      goto LABEL_20;
    }
LABEL_13:
    v22 = (_QWORD *)sub_986580(v21);
    v23 = sub_986580(v12);
    sub_B444E0(v22, v23 + 24, 0);
    v24 = sub_AA48A0(v12);
    sub_B43C20((__int64)&v52, *(_QWORD *)a1);
    v25 = sub_BD2C40(72, unk_3F148B8);
    if ( v25 )
      sub_B4C8A0((__int64)v25, v24, (__int64)v52, v53);
    v26 = (_QWORD *)sub_986580(v12);
    sub_B43D60(v26);
    goto LABEL_16;
  }
  v21 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
    goto LABEL_13;
LABEL_20:
  sub_B1AEF0(*(_QWORD *)(a1 + 32), v47, v12);
  v33 = *(_QWORD *)(a1 + 16);
  if ( v33 )
  {
    sub_B19380(*(_QWORD *)(a1 + 32), v33);
    sub_D48300(*(_QWORD *)(a1 + 40), *(_QWORD *)(a1 + 16));
  }
  if ( *(_QWORD *)a1 )
  {
    sub_B19380(*(_QWORD *)(a1 + 32), *(_QWORD *)a1);
    sub_D48300(*(_QWORD *)(a1 + 40), *(_QWORD *)a1);
  }
  result = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 1808) = *(_QWORD *)a2;
  return result;
}
