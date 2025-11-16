// Function: sub_DCFD50
// Address: 0xdcfd50
//
__int64 __fastcall sub_DCFD50(__int64 *a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  _QWORD *v7; // r14
  __int16 v8; // ax
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  __int64 result; // rax
  _QWORD *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rbx
  _QWORD *v20; // r13
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // ebx
  unsigned int v27; // eax
  __int64 v28; // rsi
  unsigned __int64 v29; // rdx
  bool v30; // zf
  __int64 *v31; // rax
  _QWORD *v32; // rax
  bool v33; // cc
  _QWORD *v34; // rax
  _QWORD *v35; // r12
  __int64 v36; // [rsp+8h] [rbp-78h] BYREF
  __int64 v37; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v40; // [rsp+28h] [rbp-58h]
  __int64 *v41; // [rsp+30h] [rbp-50h]
  __int64 *v42; // [rsp+38h] [rbp-48h]
  __int64 *v43; // [rsp+40h] [rbp-40h]

  v36 = a2;
  if ( *(_BYTE *)(sub_D95540(a2) + 8) == 14 )
    return 0;
  v7 = (_QWORD *)v36;
  v8 = *(_WORD *)(v36 + 24);
  if ( v8 == 3 )
  {
    v9 = *(_QWORD *)(v36 + 32);
    if ( *(_WORD *)(v9 + 24) != 2 )
      return 0;
    v10 = *(_QWORD *)(v9 + 32);
    *a3 = v10;
    v11 = sub_D95540(v10);
    v12 = sub_D97050((__int64)a1, v11);
    v13 = sub_D95540(v36);
    if ( v12 > sub_D97050((__int64)a1, v13) )
      return 0;
    v23 = sub_D95540(*a3);
    if ( v23 != sub_D95540(v36) )
    {
      v24 = sub_D95540(v36);
      *a3 = (__int64)sub_DC2B70((__int64)a1, *a3, v24, 0);
    }
    v25 = sub_D95540(v36);
    v38 = sub_D97050((__int64)a1, v25);
    if ( v38 > 0x40 )
      sub_C43690((__int64)&v37, 1, 0);
    else
      v37 = 1;
    v26 = sub_D97050((__int64)a1, *(_QWORD *)(v9 + 40));
    v27 = v38;
    LODWORD(v40) = v38;
    if ( v38 > 0x40 )
    {
      sub_C43780((__int64)&v39, (const void **)&v37);
      v27 = (unsigned int)v40;
      if ( (unsigned int)v40 > 0x40 )
      {
        sub_C47690(&v39, v26);
LABEL_27:
        v32 = sub_DA26C0(a1, (__int64)&v39);
        v33 = (unsigned int)v40 <= 0x40;
        *a4 = (__int64)v32;
        if ( !v33 && v39 )
          j_j___libc_free_0_0(v39);
        if ( v38 > 0x40 )
        {
          if ( v37 )
            j_j___libc_free_0_0(v37);
        }
        return 1;
      }
    }
    else
    {
      v39 = v37;
    }
    v28 = 0;
    if ( v26 != v27 )
      v28 = v39 << v26;
    v29 = v28 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v27);
    v30 = v27 == 0;
    v31 = 0;
    if ( !v30 )
      v31 = (__int64 *)v29;
    v39 = (__int64)v31;
    goto LABEL_27;
  }
  if ( v8 != 5 )
    return 0;
  if ( *(_QWORD *)(v36 + 40) != 2 )
    return 0;
  v15 = *(_QWORD **)(v36 + 32);
  v16 = v15[1];
  v37 = v16;
  v17 = *v15;
  if ( *(_WORD *)(*v15 + 24LL) != 6 )
    return 0;
  v41 = a1;
  v39 = (__int64)&v36;
  v40 = &v37;
  v42 = a3;
  v43 = a4;
  v18 = *(_QWORD *)(v17 + 40);
  if ( v18 == 3 )
  {
    v34 = *(_QWORD **)(v17 + 32);
    if ( *(_WORD *)(*v34 + 24LL) )
      return 0;
    v19 = v34[1];
    if ( v7 != sub_DCFA50(a1, v16, v19) )
    {
      v19 = *(_QWORD *)(*(_QWORD *)(v17 + 32) + 16LL);
      v35 = *(_QWORD **)v39;
      if ( v35 != sub_DCFA50(v41, *v40, v19) )
        return 0;
    }
LABEL_37:
    *v42 = *v40;
    *v43 = v19;
    return 1;
  }
  if ( v18 != 2 )
    return 0;
  v19 = *(_QWORD *)(*(_QWORD *)(v17 + 32) + 8LL);
  if ( v7 == sub_DCFA50(a1, v16, v19) )
    goto LABEL_37;
  v19 = **(_QWORD **)(v17 + 32);
  v20 = *(_QWORD **)v39;
  if ( v20 == sub_DCFA50(v41, *v40, v19) )
    goto LABEL_37;
  v21 = sub_DCAF50(a1, *(_QWORD *)(*(_QWORD *)(v17 + 32) + 8LL), 0);
  result = sub_DCFC20((_QWORD **)&v39, (__int64)v21);
  if ( !(_BYTE)result )
  {
    v22 = sub_DCAF50(a1, **(_QWORD **)(v17 + 32), 0);
    return sub_DCFC20((_QWORD **)&v39, (__int64)v22);
  }
  return result;
}
