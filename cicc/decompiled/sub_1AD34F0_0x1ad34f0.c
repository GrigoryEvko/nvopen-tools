// Function: sub_1AD34F0
// Address: 0x1ad34f0
//
void __fastcall sub_1AD34F0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // r14
  int v12; // eax
  __int64 *v13; // r8
  __int64 v14; // rsi
  __int64 *v15; // r12
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 v23; // [rsp+18h] [rbp-A8h]
  __int64 v25; // [rsp+28h] [rbp-98h]
  __int64 v26; // [rsp+30h] [rbp-90h]
  __int64 v27; // [rsp+38h] [rbp-88h]
  unsigned int v28; // [rsp+40h] [rbp-80h]
  __int64 *v29; // [rsp+40h] [rbp-80h]
  __int64 v30; // [rsp+48h] [rbp-78h]
  unsigned __int8 *v31; // [rsp+50h] [rbp-70h] BYREF
  __int64 v32; // [rsp+58h] [rbp-68h] BYREF
  unsigned __int8 *v33; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int8 *v34; // [rsp+68h] [rbp-58h] BYREF
  __int64 v35; // [rsp+70h] [rbp-50h] BYREF
  __int64 v36; // [rsp+78h] [rbp-48h]
  __int64 v37; // [rsp+80h] [rbp-40h]
  int v38; // [rsp+88h] [rbp-38h]

  if ( *(_QWORD *)(a3 + 48) )
  {
    v23 = a2;
    v5 = (__int64 *)sub_15E0530(a1);
    v6 = sub_15C70A0(a3 + 48);
    v7 = 0;
    v8 = *(unsigned int *)(v6 + 8);
    if ( (_DWORD)v8 == 2 )
      v7 = *(_QWORD *)(v6 - 8);
    v9 = sub_15B9E00(v5, *(_DWORD *)(v6 + 4), *(unsigned __int16 *)(v6 + 2), *(_QWORD *)(v6 - 8 * v8), v7, 1u, 1);
    v35 = 0;
    v25 = v9;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    if ( a2 == a1 + 72 )
    {
      v17 = 0;
      goto LABEL_36;
    }
    while ( 1 )
    {
      if ( !v23 )
        BUG();
      v10 = *(_QWORD *)(v23 + 24);
      v30 = v23 + 16;
      if ( v23 + 16 != v10 )
        break;
LABEL_34:
      v23 = *(_QWORD *)(v23 + 8);
      if ( v23 == a1 + 72 )
      {
        v17 = v36;
LABEL_36:
        j___libc_free_0(v17);
        return;
      }
    }
    while ( 1 )
    {
      if ( !v10 )
        BUG();
      v14 = *(_QWORD *)(v10 + 24);
      v31 = (unsigned __int8 *)v14;
      if ( v14 )
      {
        sub_1623A60((__int64)&v31, v14, 2);
        if ( v31 )
        {
          v11 = (__int64 *)sub_16498A0(v10 - 24);
          v34 = v31;
          if ( v31 )
            sub_1623A60((__int64)&v34, (__int64)v31, 2);
          sub_15C7550(&v32, (__int64)&v34, v25, v11, (__int64)&v35, 0);
          if ( v34 )
            sub_161E7C0((__int64)&v34, (__int64)v34);
          v26 = sub_15C70A0((__int64)&v32);
          v27 = sub_15C70D0((__int64)&v31);
          v28 = sub_15C70C0((__int64)&v31);
          v12 = sub_15C70B0((__int64)&v31);
          sub_15C7110(&v33, v12, v28, v27, v26);
          v13 = (__int64 *)(v10 + 24);
          v34 = v33;
          if ( v33 )
          {
            sub_1623A60((__int64)&v34, (__int64)v33, 2);
            v13 = (__int64 *)(v10 + 24);
            if ( (unsigned __int8 **)(v10 + 24) == &v34 )
            {
              if ( v34 )
                sub_161E7C0((__int64)&v34, (__int64)v34);
              goto LABEL_17;
            }
            v18 = *(_QWORD *)(v10 + 24);
            if ( !v18 )
            {
LABEL_41:
              v19 = v34;
              *(_QWORD *)(v10 + 24) = v34;
              if ( v19 )
                sub_1623210((__int64)&v34, v19, (__int64)v13);
              goto LABEL_17;
            }
          }
          else if ( v13 == (__int64 *)&v34 || (v18 = *(_QWORD *)(v10 + 24)) == 0 )
          {
LABEL_17:
            if ( v33 )
              sub_161E7C0((__int64)&v33, (__int64)v33);
            if ( v32 )
              sub_161E7C0((__int64)&v32, v32);
            if ( v31 )
              sub_161E7C0((__int64)&v31, (__int64)v31);
            goto LABEL_23;
          }
          v29 = v13;
          sub_161E7C0((__int64)v13, v18);
          v13 = v29;
          goto LABEL_41;
        }
      }
      if ( a4
        || *(_BYTE *)(v10 - 8) == 53
        && *(_BYTE *)(*(_QWORD *)(v10 - 48) + 16LL) <= 0x10u
        && (*(_BYTE *)(v10 - 6) & 0x20) == 0 )
      {
        goto LABEL_23;
      }
      v15 = (__int64 *)(v10 + 24);
      v16 = *(_QWORD *)(a3 + 48);
      v34 = (unsigned __int8 *)v16;
      if ( !v16 )
      {
        if ( v15 == (__int64 *)&v34 )
          goto LABEL_23;
        v20 = *(_QWORD *)(v10 + 24);
        if ( !v20 )
          goto LABEL_23;
        goto LABEL_47;
      }
      sub_1623A60((__int64)&v34, v16, 2);
      if ( v15 != (__int64 *)&v34 )
      {
        v20 = *(_QWORD *)(v10 + 24);
        if ( !v20 )
        {
LABEL_48:
          v21 = v34;
          *(_QWORD *)(v10 + 24) = v34;
          if ( v21 )
            sub_1623210((__int64)&v34, v21, v10 + 24);
          goto LABEL_23;
        }
LABEL_47:
        sub_161E7C0(v10 + 24, v20);
        goto LABEL_48;
      }
      if ( !v34 )
      {
LABEL_23:
        v10 = *(_QWORD *)(v10 + 8);
        if ( v30 == v10 )
          goto LABEL_34;
      }
      else
      {
        sub_161E7C0((__int64)&v34, (__int64)v34);
        v10 = *(_QWORD *)(v10 + 8);
        if ( v30 == v10 )
          goto LABEL_34;
      }
    }
  }
}
