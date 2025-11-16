// Function: sub_2CB1590
// Address: 0x2cb1590
//
__int64 __fastcall sub_2CB1590(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, int *a5, unsigned int a6)
{
  __int64 **v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 *v12; // r8
  _BYTE *v13; // rcx
  unsigned int v14; // r13d
  _BYTE *v15; // r15
  unsigned int v16; // ebx
  __int64 *v17; // r14
  __int64 v18; // rdx
  __int64 *v19; // rbx
  __int64 v20; // r14
  __int64 v21; // rdi
  int v22; // esi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rbx
  bool v30; // al
  bool v31; // al
  __int64 v32; // rdx
  char v33; // al
  _QWORD *v34; // rax
  __int64 v35; // r12
  bool v36; // zf
  __int64 *v38; // [rsp+10h] [rbp-E0h]
  __int64 *v39; // [rsp+20h] [rbp-D0h]
  __int64 v40; // [rsp+28h] [rbp-C8h]
  _BYTE *v41; // [rsp+38h] [rbp-B8h]
  __int64 v42; // [rsp+38h] [rbp-B8h]
  __int64 *v45; // [rsp+50h] [rbp-A0h]
  __int64 v46; // [rsp+58h] [rbp-98h]
  __int64 v47; // [rsp+60h] [rbp-90h]
  __int64 v49; // [rsp+70h] [rbp-80h]
  unsigned int v50; // [rsp+78h] [rbp-78h]
  __int64 v51; // [rsp+78h] [rbp-78h]
  __int64 v52; // [rsp+78h] [rbp-78h]
  __int64 v53; // [rsp+88h] [rbp-68h] BYREF
  _QWORD v54[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v55; // [rsp+B0h] [rbp-40h]

  v45 = (__int64 *)sub_BCCE00(a1, a6);
  v50 = 0x20 / a6;
  v9 = (__int64 **)sub_BCDA70(v45, 0x20 / a6);
  v10 = sub_ACA8A0(v9);
  v11 = sub_BCCE00(a1, 0x20u);
  v12 = *(__int64 **)a4;
  v40 = v11;
  v47 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( *(_QWORD *)a4 == v47 )
  {
LABEL_16:
    v54[0] = "vec";
    v55 = 259;
    v27 = sub_BD2C40(72, 1u);
    v28 = (__int64)v27;
    if ( v27 )
      sub_B51BF0((__int64)v27, v10, v40, (__int64)v54, 0, 0);
    sub_B43DD0(v28, *a3);
    *a3 = v28;
  }
  else
  {
    v53 = 0;
    v39 = v12;
    v13 = sub_2CB1240(a2, (_BYTE *)*v12, &v53);
    v38 = a3;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = v39;
    while ( v16 >> 3 == v53 )
    {
      if ( !v15 )
      {
        v41 = v13;
        v54[0] = sub_9208B0(a2, *((_QWORD *)v13 + 1));
        v54[1] = v18;
        if ( sub_CA1930(v54) != 32 )
        {
LABEL_10:
          a3 = v38;
          goto LABEL_11;
        }
        v15 = v41;
      }
      ++v17;
      ++v14;
      if ( (__int64 *)v47 == v17 || v50 <= v14 )
      {
        v28 = (__int64)v15;
        v36 = v50 == v14;
        a3 = v38;
        v31 = v36;
        goto LABEL_21;
      }
      v53 = 0;
      v16 += a6;
      v13 = sub_2CB1240(a2, (_BYTE *)*v17, &v53);
      if ( v13 != v15 )
        goto LABEL_10;
    }
    v28 = (__int64)v15;
    v30 = v50 == v14;
    a3 = v38;
    v31 = v15 != 0 && v30;
LABEL_21:
    if ( !v31
      || (v32 = *(_QWORD *)(v28 + 8), v33 = *(_BYTE *)(v32 + 8), v33 == 7 || v33 == 13)
      || (unsigned __int8)(v33 - 15) <= 1u )
    {
LABEL_11:
      v19 = *(__int64 **)a4;
      v42 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
      if ( *(_QWORD *)a4 != v42 )
      {
        v20 = 0;
        do
        {
          v21 = *v19;
          v54[0] = "vec.elem";
          v22 = *a5;
          v55 = 259;
          v23 = sub_2CAF330(v21, v22, (__int64)v45, a3, (__int64)v54);
          v55 = 257;
          v51 = v20++;
          v46 = v23;
          v24 = sub_BCCE00(a1, 0x20u);
          v25 = sub_ACD640(v24, v51, 0);
          v52 = v10;
          v49 = v25;
          v26 = sub_BD2C40(72, 3u);
          v10 = (__int64)v26;
          if ( v26 )
            sub_B4DFA0((__int64)v26, v52, v46, v49, (__int64)v54, v46, 0, 0);
          ++v19;
          sub_B43DD0(v10, *a3);
          *a3 = v10;
        }
        while ( v19 != (__int64 *)v42 );
      }
      goto LABEL_16;
    }
    if ( v40 != v32 )
    {
      v54[0] = "base.bitcast";
      v55 = 259;
      v34 = sub_BD2C40(72, 1u);
      v35 = (__int64)v34;
      if ( v34 )
        sub_B51BF0((__int64)v34, v28, v40, (__int64)v54, 0, 0);
      v28 = v35;
      sub_B43DD0(v35, *a3);
      *a3 = v35;
    }
  }
  return v28;
}
