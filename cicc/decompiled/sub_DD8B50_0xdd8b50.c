// Function: sub_DD8B50
// Address: 0xdd8b50
//
_QWORD *__fastcall sub_DD8B50(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rbx
  __int64 v18; // rsi
  unsigned int v19; // ebx
  _QWORD *v20; // r14
  __int64 v21; // rcx
  __int64 v22; // r8
  __int16 v23; // r12
  int v24; // eax
  _QWORD *v25; // rax
  int v27; // eax
  int v28; // r10d
  __int64 *v29; // [rsp+8h] [rbp-78h]
  __int64 *v30; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v31; // [rsp+18h] [rbp-68h]
  int v32; // [rsp+20h] [rbp-60h] BYREF
  __int64 v33; // [rsp+28h] [rbp-58h]
  __int64 v34; // [rsp+30h] [rbp-50h]
  char v35; // [rsp+38h] [rbp-48h]
  char v36; // [rsp+39h] [rbp-47h]
  char v37; // [rsp+48h] [rbp-38h]

  v9 = *(_QWORD *)(a1 + 48);
  v10 = *(_QWORD *)(a2 + 40);
  v11 = *(unsigned int *)(v9 + 24);
  v12 = *(_QWORD *)(v9 + 8);
  if ( !(_DWORD)v11 )
  {
LABEL_22:
    v31 = 0;
    goto LABEL_4;
  }
  v11 = (unsigned int)(v11 - 1);
  v13 = v11 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  a5 = *v14;
  if ( v10 != *v14 )
  {
    v27 = 1;
    while ( a5 != -4096 )
    {
      v28 = v27 + 1;
      v13 = v11 & (v27 + v13);
      v14 = (__int64 *)(v12 + 16LL * v13);
      a5 = *v14;
      if ( v10 == *v14 )
        goto LABEL_3;
      v27 = v28;
    }
    goto LABEL_22;
  }
LABEL_3:
  v31 = (unsigned __int8 *)v14[1];
LABEL_4:
  sub_D94080((__int64)&v32, a3, *(__int64 **)(a1 + 40), v11, a5);
  if ( !v37 || v32 != 13 )
    return 0;
  v17 = v33;
  v18 = v34;
  if ( a2 != v33 || (v18 = v34, !(unsigned __int8)sub_D48480((__int64)v31, v34, v15, v16)) )
  {
    if ( v18 != a2 || !(unsigned __int8)sub_D48480((__int64)v31, v17, v15, v16) )
      return 0;
    v18 = v17;
  }
  v30 = sub_DD8400(a1, v18);
  if ( !v30 )
    return 0;
  v19 = 2 * (v36 != 0);
  if ( v35 )
    v19 |= 4u;
  v29 = sub_DD8400(a1, a4);
  v20 = sub_DC1960(a1, (__int64)v29, (__int64)v30, (__int64)v31, v19);
  sub_DB77A0(a1, a2, (__int64)v20);
  if ( *((_WORD *)v20 + 12) == 8 )
  {
    v23 = *((_WORD *)v20 + 14);
    v24 = sub_DCF420((__int64 *)a1, (__int64)v20);
    sub_D97270(a1, (__int64)v20, v23 & 7 | v24);
  }
  if ( *a3 > 0x1Cu && (unsigned __int8)sub_DD8750((__int64 *)a1, (__int64)a3, v31, v21, v22) )
  {
    v25 = sub_DC7ED0((__int64 *)a1, (__int64)v29, (__int64)v30, 0, 0);
    sub_DC1960(a1, (__int64)v25, (__int64)v30, (__int64)v31, v19);
  }
  return v20;
}
