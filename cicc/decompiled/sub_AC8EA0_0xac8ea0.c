// Function: sub_AC8EA0
// Address: 0xac8ea0
//
__int64 __fastcall sub_AC8EA0(__int64 *a1, __int64 *a2)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 *v7; // r12
  __int64 result; // rax
  __int64 *v9; // rsi
  int v10; // eax
  int v11; // eax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 *v18; // r12
  __int64 v19; // r14
  __int64 v20; // r13
  char v21; // al
  __int64 v22; // [rsp+8h] [rbp-68h]
  __int64 *v23; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v24; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v25[10]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *a1;
  v5 = *a1 + 336;
  v6 = v5;
  if ( !(unsigned __int8)sub_AC67B0(v5, a2, &v23) )
  {
    v9 = (__int64 *)*(unsigned int *)(v4 + 360);
    v24 = v23;
    v10 = *(_DWORD *)(v4 + 352);
    ++*(_QWORD *)(v4 + 336);
    v11 = v10 + 1;
    if ( 4 * v11 >= (unsigned int)(3 * (_DWORD)v9) )
    {
      LODWORD(v9) = 2 * (_DWORD)v9;
    }
    else if ( (int)v9 - *(_DWORD *)(v4 + 356) - v11 > (unsigned int)v9 >> 3 )
    {
      goto LABEL_6;
    }
    sub_AC8970(v5, (int)v9);
    v9 = a2;
    v6 = v5;
    sub_AC67B0(v5, a2, &v24);
    v11 = *(_DWORD *)(v4 + 352) + 1;
LABEL_6:
    *(_DWORD *)(v4 + 352) = v11;
    v12 = sub_C33690();
    v16 = sub_C33340(v6, v9, v13, v14, v15);
    v17 = v16;
    if ( v12 == v16 )
      sub_C3C5A0(v25, v16, 1);
    else
      sub_C36740(v25, v12, 1);
    v18 = v24;
    if ( *v24 != v25[0] || (*v24 == v17 ? (v21 = sub_C3E590(v24)) : (v21 = sub_C33D00(v24)), v18 = v24, !v21) )
      --*(_DWORD *)(v4 + 356);
    sub_91D830(v25);
    if ( *v18 == v17 )
    {
      if ( *a2 == v17 )
      {
        sub_C3C9E0(v18, a2);
        goto LABEL_13;
      }
    }
    else if ( *a2 != v17 )
    {
      sub_C33E70(v18, a2);
      goto LABEL_13;
    }
    if ( a2 != v18 )
    {
      sub_91D830(v18);
      if ( v17 != *a2 )
      {
        sub_C33EB0(v18, a2);
        v7 = v18 + 3;
        *v7 = 0;
LABEL_14:
        result = *v7;
        if ( *v7 )
          return result;
        goto LABEL_15;
      }
      sub_C3C790(v18, a2);
    }
LABEL_13:
    v18[3] = 0;
    v7 = v18 + 3;
    goto LABEL_14;
  }
  v7 = v23 + 3;
  result = v23[3];
  if ( result )
    return result;
LABEL_15:
  v19 = sub_BCB1D0(a1, *a2);
  result = sub_BD2C40(48, unk_3F289A4);
  if ( result )
  {
    v22 = result;
    sub_AC3040(result, v19, a2);
    result = v22;
  }
  v20 = *v7;
  *v7 = result;
  if ( v20 )
  {
    sub_91D830((_QWORD *)(v20 + 24));
    sub_BD7260(v20);
    sub_BD2DD0(v20);
    return *v7;
  }
  return result;
}
