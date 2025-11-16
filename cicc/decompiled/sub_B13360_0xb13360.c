// Function: sub_B13360
// Address: 0xb13360
//
__int64 **__fastcall sub_B13360(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, char a4)
{
  char v6; // r12
  __int64 *v7; // rcx
  __int64 **result; // rax
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 **v12; // r9
  _BYTE *v13; // rsi
  unsigned __int8 *v14; // rbx
  __int64 v15; // r8
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // r10
  __int64 v19; // r11
  _QWORD *v20; // r8
  _QWORD *v21; // rax
  __int64 v22; // r11
  __int64 v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 *v30; // r15
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // rbx
  _QWORD *v35; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+8h] [rbp-A8h]
  _QWORD *v37; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v38; // [rsp+10h] [rbp-A0h]
  __int64 v39; // [rsp+10h] [rbp-A0h]
  __int64 **v41; // [rsp+18h] [rbp-98h]
  __int64 **v42; // [rsp+18h] [rbp-98h]
  __int64 **v43; // [rsp+40h] [rbp-70h] BYREF
  __int64 **v44; // [rsp+48h] [rbp-68h]
  __int64 *v45; // [rsp+50h] [rbp-60h] BYREF
  __int64 v46; // [rsp+58h] [rbp-58h]
  _BYTE v47[80]; // [rsp+60h] [rbp-50h] BYREF

  if ( *(_BYTE *)(a1 + 64) == 2 && a2 == sub_B13320(a1) )
  {
    v27 = sub_B98A20(a3, a2, v25, v26);
    sub_B91340(a1 + 40, 1);
    *(_QWORD *)(a1 + 48) = v27;
    v6 = 1;
    sub_B96F50(a1 + 40, 1);
  }
  else
  {
    v6 = 0;
  }
  sub_B129C0(&v43, a1);
  result = v43;
  v9 = (__int64)v43;
  v45 = (__int64 *)v43;
  if ( v44 == v43 )
  {
LABEL_19:
    if ( !v6 && !a4 )
      BUG();
    return result;
  }
  while ( 1 )
  {
    result = (__int64 **)(v9 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v9 & 4) != 0 )
    {
      v7 = *result;
      if ( a2 == (unsigned __int8 *)(*result)[17] )
        break;
      goto LABEL_6;
    }
    if ( a2 == (unsigned __int8 *)result[17] )
      break;
    if ( result )
    {
      result += 18;
      v9 = (__int64)result;
      if ( result == v44 )
        goto LABEL_19;
    }
    else
    {
LABEL_6:
      result = (__int64 **)((unsigned __int64)(result + 1) | 4);
      v9 = (__int64)result;
      if ( result == v44 )
        goto LABEL_19;
    }
  }
  v12 = v44;
  v13 = (_BYTE *)v9;
  if ( v44 == (__int64 **)v9 )
    goto LABEL_19;
  v10 = *a3;
  if ( **(_BYTE **)(a1 + 40) != 4 )
  {
    if ( (_BYTE)v10 == 24 )
      v11 = *((_QWORD *)a3 + 3);
    else
      v11 = sub_B98A20(a3, v9, v10, v7);
    sub_B91340(a1 + 40, 0);
    *(_QWORD *)(a1 + 40) = v11;
    return (__int64 **)sub_B96F50(a1 + 40, 0);
  }
  v45 = (__int64 *)v47;
  v46 = 0x400000000LL;
  if ( (_BYTE)v10 == 24 )
  {
    v14 = (unsigned __int8 *)*((_QWORD *)a3 + 3);
    if ( (unsigned int)*v14 - 1 >= 2 )
      v14 = 0;
  }
  else
  {
    v28 = sub_B98A20(a3, v9, v10, v7);
    v12 = v44;
    v14 = (unsigned __int8 *)v28;
  }
  v15 = (__int64)v43;
  if ( v12 != v43 )
  {
    while ( 2 )
    {
      v19 = v15;
      v20 = (_QWORD *)(v15 & 0xFFFFFFFFFFFFFFF8LL);
      v21 = v20;
      v22 = (v19 >> 2) & 1;
      if ( (_DWORD)v22 )
        v21 = (_QWORD *)*v20;
      v23 = v21[17];
      v24 = (_QWORD *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
      if ( ((v9 >> 2) & 1) != 0 )
        v24 = (_QWORD *)*v24;
      if ( v24[17] == v23 )
      {
        v17 = (unsigned int)v46;
        v16 = v14;
        v18 = (unsigned int)v46 + 1LL;
        if ( v18 > HIDWORD(v46) )
          goto LABEL_40;
      }
      else
      {
        if ( *(_BYTE *)v23 == 24 )
        {
          v16 = *(unsigned __int8 **)(v23 + 24);
          v13 = 0;
          if ( (unsigned int)*v16 - 1 > 1 )
            v16 = 0;
        }
        else
        {
          v37 = v20;
          v39 = (unsigned int)v22;
          v42 = v12;
          v16 = (unsigned __int8 *)sub_B98A20(v23, v13, v10, v7);
          v20 = v37;
          v22 = v39;
          v12 = v42;
        }
        v17 = (unsigned int)v46;
        v18 = (unsigned int)v46 + 1LL;
        if ( v18 > HIDWORD(v46) )
        {
LABEL_40:
          v13 = v47;
          v35 = v20;
          v36 = v22;
          v38 = v16;
          v41 = v12;
          sub_C8D5F0(&v45, v47, v18, 8);
          v17 = (unsigned int)v46;
          v20 = v35;
          v22 = v36;
          v16 = v38;
          v12 = v41;
        }
      }
      v7 = v45;
      v45[v17] = (__int64)v16;
      v10 = (unsigned int)(v46 + 1);
      LODWORD(v46) = v46 + 1;
      if ( v22 || !v20 )
      {
        v15 = (unsigned __int64)(v20 + 1) | 4;
        if ( v12 == (__int64 **)v15 )
          goto LABEL_45;
      }
      else
      {
        v15 = (__int64)(v20 + 18);
        if ( v12 == (__int64 **)v15 )
          goto LABEL_45;
      }
      continue;
    }
  }
  LODWORD(v10) = v46;
LABEL_45:
  v29 = (unsigned int)v10;
  v30 = v45;
  v31 = sub_B12A50(a1, 0);
  v33 = (__int64 *)sub_BD5C60(v31, 0, v32);
  v34 = sub_B00B60(v33, v30, v29);
  sub_B91340(a1 + 40, 0);
  *(_QWORD *)(a1 + 40) = v34;
  result = (__int64 **)sub_B96F50(a1 + 40, 0);
  if ( v45 != (__int64 *)v47 )
    return (__int64 **)_libc_free(v45, 0);
  return result;
}
