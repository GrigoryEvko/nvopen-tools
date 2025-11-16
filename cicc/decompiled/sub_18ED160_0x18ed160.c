// Function: sub_18ED160
// Address: 0x18ed160
//
__int64 __fastcall sub_18ED160(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  unsigned __int64 v4; // rdi
  __int64 *v5; // rdx
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 v9; // r12
  __int64 *v10; // r13
  __int64 v11; // rsi
  __int64 *v12; // r12
  __int64 v13; // rbx
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 **v22; // [rsp+10h] [rbp-100h]
  __int64 *v23; // [rsp+18h] [rbp-F8h]
  __int64 *v24; // [rsp+20h] [rbp-F0h]
  __int64 **v25; // [rsp+28h] [rbp-E8h]
  __int64 *v26; // [rsp+30h] [rbp-E0h]
  __int64 v27; // [rsp+38h] [rbp-D8h]
  __int64 *v28; // [rsp+40h] [rbp-D0h]
  __int64 v29; // [rsp+48h] [rbp-C8h]
  __int64 v30[2]; // [rsp+50h] [rbp-C0h] BYREF
  char v31; // [rsp+60h] [rbp-B0h]
  char v32; // [rsp+61h] [rbp-AFh]
  char v33[8]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v34; // [rsp+78h] [rbp-98h]
  __int64 *v35; // [rsp+80h] [rbp-90h]
  int v36; // [rsp+88h] [rbp-88h]
  int v37; // [rsp+8Ch] [rbp-84h]
  int v38; // [rsp+90h] [rbp-80h]

  v1 = *(_QWORD *)(a1 + 136);
  v2 = 632LL * *(unsigned int *)(a1 + 144);
  v22 = (__int64 **)(v1 + v2);
  if ( v1 == v1 + v2 )
    return 0;
  v25 = *(__int64 ***)(a1 + 136);
  do
  {
    sub_18E9800((__int64)v33, (_QWORD *)a1, (__int64)v25);
    v4 = (unsigned __int64)v35;
    v5 = v34;
    if ( v35 == v34 )
      v24 = &v35[v37];
    else
      v24 = &v35[v36];
    if ( v35 != v24 )
    {
      v6 = v35;
      while ( 1 )
      {
        v7 = *v6;
        if ( (unsigned __int64)*v6 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v24 == ++v6 )
          goto LABEL_9;
      }
      v23 = v6;
      if ( v6 != v24 )
      {
        while ( 1 )
        {
          v9 = **v25;
          v32 = 1;
          v30[0] = (__int64)"const";
          v31 = 3;
          v10 = sub_1648A60(56, 1u);
          if ( v10 )
            sub_15FD590((__int64)v10, (__int64)*v25, v9, (__int64)v30, v7);
          v11 = *(_QWORD *)(v7 + 48);
          v12 = v10 + 6;
          v30[0] = v11;
          if ( !v11 )
            break;
          sub_1623A60((__int64)v30, v11, 2);
          if ( v12 == v30 )
          {
            if ( v30[0] )
              sub_161E7C0((__int64)v30, v30[0]);
            goto LABEL_20;
          }
          v20 = v10[6];
          if ( v20 )
            goto LABEL_43;
LABEL_44:
          v21 = (unsigned __int8 *)v30[0];
          v10[6] = v30[0];
          if ( v21 )
            sub_1623210((__int64)v30, v21, (__int64)(v10 + 6));
LABEL_20:
          v28 = v25[1];
          v26 = &v28[19 * *((unsigned int *)v25 + 4)];
          if ( v28 != v26 )
          {
            while ( 1 )
            {
              v13 = *v28;
              v27 = *v28 + 16LL * *((unsigned int *)v28 + 2);
              if ( *v28 != v27 )
                break;
LABEL_34:
              v28 += 19;
              if ( v26 == v28 )
                goto LABEL_35;
            }
            while ( 1 )
            {
LABEL_28:
              v16 = sub_18E8200(a1, *(_QWORD *)v13, *(_DWORD *)(v13 + 8));
              if ( v37 - v38 == 1 || sub_15CC8F0(*(_QWORD *)(a1 + 8), v10[5], *(_QWORD *)(v16 + 40)) )
                sub_18EC9E0(a1, v10, v28[18], v13);
              v29 = sub_15C70A0(*(_QWORD *)v13 + 48LL);
              v17 = sub_15C70A0((__int64)(v10 + 6));
              v18 = sub_15BA070(v17, v29, 0);
              sub_15C7080(v30, v18);
              if ( v12 != v30 )
                break;
              if ( !v30[0] )
                goto LABEL_27;
              v13 += 16;
              sub_161E7C0((__int64)(v10 + 6), v30[0]);
              if ( v27 == v13 )
                goto LABEL_34;
            }
            v14 = v10[6];
            if ( v14 )
              sub_161E7C0((__int64)(v10 + 6), v14);
            v15 = (unsigned __int8 *)v30[0];
            v10[6] = v30[0];
            if ( v15 )
              sub_1623210((__int64)v30, v15, (__int64)(v10 + 6));
LABEL_27:
            v13 += 16;
            if ( v27 == v13 )
              goto LABEL_34;
            goto LABEL_28;
          }
LABEL_35:
          v19 = v23 + 1;
          if ( v23 + 1 != v24 )
          {
            while ( 1 )
            {
              v7 = *v19;
              if ( (unsigned __int64)*v19 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v24 == ++v19 )
                goto LABEL_38;
            }
            v23 = v19;
            if ( v19 != v24 )
              continue;
          }
LABEL_38:
          v4 = (unsigned __int64)v35;
          v5 = v34;
          goto LABEL_9;
        }
        if ( v12 == v30 )
          goto LABEL_20;
        v20 = v10[6];
        if ( !v20 )
          goto LABEL_20;
LABEL_43:
        sub_161E7C0((__int64)(v10 + 6), v20);
        goto LABEL_44;
      }
    }
LABEL_9:
    if ( v5 != (__int64 *)v4 )
      _libc_free(v4);
    v25 += 79;
  }
  while ( v22 != v25 );
  return 1;
}
