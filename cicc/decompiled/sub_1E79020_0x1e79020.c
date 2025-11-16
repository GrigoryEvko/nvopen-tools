// Function: sub_1E79020
// Address: 0x1e79020
//
void __fastcall sub_1E79020(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // r15
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  __int64 *v11; // rbx
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 *i; // r15
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // r14
  __int64 v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 *v28; // [rsp+30h] [rbp-50h]
  __int64 v29; // [rsp+38h] [rbp-48h] BYREF
  __int64 v30[7]; // [rsp+48h] [rbp-38h] BYREF

  v29 = a3;
  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v28 = a1 + 1;
    while ( 1 )
    {
      v12 = v29;
      v13 = *(_QWORD *)(v29 + 280);
      v6 = *a1;
      v5 = *v28;
      if ( !v13 )
        break;
      v23 = *a1;
      v25 = *v28;
      v3 = sub_1DDC3C0(v13, *v28);
      v4 = *(_QWORD *)(v12 + 280);
      v5 = v25;
      v6 = v23;
      v7 = v3;
      if ( v4 && (v8 = v23, v24 = v25, v26 = v6, v9 = sub_1DDC3C0(v4, v8), v6 = v26, v5 = v24, v7) && v9 )
      {
        v10 = *v28;
        if ( v7 < v9 )
          goto LABEL_8;
LABEL_14:
        v30[0] = v12;
        for ( i = v28; ; i[1] = v19 )
        {
          v20 = *(_QWORD *)(v12 + 280);
          v21 = *(i - 1);
          if ( !v20 )
            break;
          v15 = sub_1DDC3C0(v20, v10);
          v16 = *(_QWORD *)(v12 + 280);
          v17 = v15;
          if ( !v16 )
            break;
          v18 = sub_1DDC3C0(v16, v21);
          if ( !v17 || !v18 )
            break;
          if ( v17 >= v18 )
            goto LABEL_22;
LABEL_19:
          v19 = *--i;
        }
        if ( sub_1E78020((__int64)v30, v10, v21) )
          goto LABEL_19;
LABEL_22:
        *i = v10;
        if ( ++v28 == a2 )
          return;
      }
      else
      {
        v10 = *v28;
LABEL_13:
        if ( !sub_1E78020((__int64)&v29, v5, v6) )
          goto LABEL_14;
LABEL_8:
        v11 = v28 + 1;
        if ( a1 != v28 )
          memmove(a1 + 1, a1, (char *)v28 - (char *)a1);
        ++v28;
        *a1 = v10;
        if ( v11 == a2 )
          return;
      }
    }
    v10 = *v28;
    goto LABEL_13;
  }
}
