// Function: sub_3037DB0
// Address: 0x3037db0
//
__int64 __fastcall sub_3037DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int *a6, int a7, char a8)
{
  __int64 v8; // r10
  __int64 v9; // r15
  __int64 v11; // rcx
  __int64 result; // rax
  bool v15; // al
  __int64 v16; // rsi
  __int64 v17; // r15
  __int64 *v18; // rsi
  char v19; // al
  int v20; // esi
  __int128 v21; // rax
  int v22; // r9d
  __int128 v23; // rax
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // r12
  __int128 v29; // rax
  int v30; // r9d
  __int128 v31; // [rsp-20h] [rbp-80h]
  __int128 v32; // [rsp-20h] [rbp-80h]
  __int128 v33; // [rsp-20h] [rbp-80h]
  int *v34; // [rsp+8h] [rbp-58h]
  int *v35; // [rsp+8h] [rbp-58h]
  int *v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  int v38; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  unsigned __int16 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+20h] [rbp-40h] BYREF
  int v43; // [rsp+28h] [rbp-38h]

  v8 = a1;
  v9 = (unsigned int)a3;
  v11 = a2;
  if ( a5 != 1 )
  {
    if ( a5 != -1 )
      return 0;
    v34 = a6;
    v15 = sub_3037D40(a1);
    v8 = a1;
    v11 = a2;
    a6 = v34;
    if ( v15 )
      return 0;
  }
  if ( *a6 == -1 )
    *a6 = 0;
  v16 = *(_QWORD *)(v11 + 80);
  v42 = v16;
  if ( v16 )
  {
    v35 = a6;
    v37 = v8;
    v39 = v11;
    sub_B96E90((__int64)&v42, v16, 1);
    a6 = v35;
    v8 = v37;
    v11 = v39;
  }
  v17 = *(_QWORD *)(v11 + 48) + 16 * v9;
  v18 = *(__int64 **)(a4 + 40);
  v36 = a6;
  v43 = *(_DWORD *)(v11 + 72);
  v40 = *(_WORD *)v17;
  v38 = *(_QWORD *)(v17 + 8);
  v19 = sub_3037D80(v8, v18);
  if ( a8 || *v36 > 0 )
  {
    if ( v40 == 12 )
    {
      v20 = v19 == 0 ? 9412 : 9414;
LABEL_14:
      *(_QWORD *)&v21 = sub_3400BD0(a4, v20, (unsigned int)&v42, 7, 0, 0, 0);
LABEL_15:
      *((_QWORD *)&v31 + 1) = a3;
      *(_QWORD *)&v31 = a2;
      result = sub_3406EB0(a4, 46, (unsigned int)&v42, v40, v38, v22, v21, v31);
      goto LABEL_16;
    }
    if ( v40 == 13 )
    {
      *(_QWORD *)&v21 = sub_3400BD0(a4, 9411, (unsigned int)&v42, 7, 0, 0, 0);
      goto LABEL_15;
    }
    result = 0;
  }
  else
  {
    if ( v40 == 12 )
    {
      v20 = 9531 - ((v19 == 0) - 1);
      goto LABEL_14;
    }
    *(_QWORD *)&v23 = sub_3400BD0(a4, 9411, (unsigned int)&v42, 7, 0, 0, 0);
    *((_QWORD *)&v32 + 1) = a3;
    *(_QWORD *)&v32 = a2;
    v25 = sub_3406EB0(a4, 46, (unsigned int)&v42, v40, v38, v24, v23, v32);
    v27 = v26;
    v28 = v25;
    *(_QWORD *)&v29 = sub_3400BD0(a4, 9262, (unsigned int)&v42, 7, 0, 0, 0);
    *((_QWORD *)&v33 + 1) = v27;
    *(_QWORD *)&v33 = v28;
    result = sub_3406EB0(a4, 46, (unsigned int)&v42, v40, v38, v30, v29, v33);
  }
LABEL_16:
  if ( v42 )
  {
    v41 = result;
    sub_B91220((__int64)&v42, v42);
    return v41;
  }
  return result;
}
