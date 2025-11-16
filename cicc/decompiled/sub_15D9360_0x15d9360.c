// Function: sub_15D9360
// Address: 0x15d9360
//
void __fastcall sub_15D9360(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  _QWORD *v13; // r15
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rbx
  _QWORD *v22; // r14
  unsigned __int64 v23; // rdi
  __int64 *v24; // r15
  __int64 v25; // r13
  __int64 *v26; // rax
  char *v27; // [rsp+8h] [rbp-B8h]
  __int64 *v28; // [rsp+18h] [rbp-A8h]
  __int64 v30; // [rsp+28h] [rbp-98h]
  __int64 v31; // [rsp+28h] [rbp-98h]
  __int64 v32; // [rsp+38h] [rbp-88h] BYREF
  __int64 v33; // [rsp+40h] [rbp-80h] BYREF
  __int64 v34; // [rsp+48h] [rbp-78h]
  char v35; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v36[8]; // [rsp+58h] [rbp-68h] BYREF
  _QWORD *v37; // [rsp+60h] [rbp-60h]
  unsigned int v38; // [rsp+70h] [rbp-50h]

  v7 = sub_15CC960(a1, a3);
  if ( v7 )
  {
    v8 = (__int64 *)v7;
    v9 = (__int64 *)sub_15CC960(a1, a4);
    if ( v9 )
    {
      v10 = sub_15CC9E0(a1, a3, a4);
      if ( v9 != (__int64 *)sub_15CC960(a1, v10) )
      {
        *(_BYTE *)(a1 + 96) = 0;
        if ( v8 == (__int64 *)v9[1] )
        {
          sub_15CF8B0((__int64)&v33, *v9, a2);
          v27 = (char *)v33;
          v28 = (__int64 *)(v33 + 8LL * (unsigned int)v34);
          if ( (__int64 *)v33 == v28 )
          {
LABEL_25:
            if ( v27 != &v35 )
              _libc_free((unsigned __int64)v27);
            v33 = *v9;
            sub_15CDD90(a1, &v33);
            v26 = (__int64 *)sub_15CC960(a1, 0);
            sub_15D8000(a1, a2, v26, v9);
            goto LABEL_18;
          }
          v24 = (__int64 *)v33;
          while ( 1 )
          {
            v31 = *v24;
            if ( sub_15CC960(a1, *v24) )
            {
              v25 = *v9;
              if ( v25 != sub_15CC9E0(a1, *v9, v31) )
                break;
            }
            if ( v28 == ++v24 )
              goto LABEL_25;
          }
          if ( v27 != &v35 )
            _libc_free((unsigned __int64)v27);
        }
        v11 = sub_15CC9E0(a1, *v8, *v9);
        v12 = sub_15CC960(a1, v11);
        v13 = *(_QWORD **)(v12 + 8);
        if ( v13 )
        {
          v14 = *(_DWORD *)(v12 + 16);
          sub_15CDF00((__int64)&v33, a2);
          sub_15D6470((__int64)&v33, v11, 0, v14, a1, 0);
          sub_15D4AE0(&v33, a1, v14);
          *(_QWORD *)(sub_15D4720((__int64)v36, (__int64 *)(v33 + 8)) + 32) = *v13;
          v15 = v33;
          v30 = (v34 - v33) >> 3;
          if ( v30 != 1 )
          {
            v16 = 1;
            while ( 1 )
            {
              v17 = *(_QWORD *)(v15 + 8 * v16++);
              v32 = v17;
              v18 = sub_15CC960(a1, v17);
              v19 = sub_15D4720((__int64)v36, &v32);
              v20 = sub_15CC960(a1, *(_QWORD *)(v19 + 32));
              sub_15CE4D0(v18, v20);
              if ( v30 == v16 )
                break;
              v15 = v33;
            }
          }
          if ( v38 )
          {
            v21 = v37;
            v22 = &v37[9 * v38];
            do
            {
              if ( *v21 != -16 && *v21 != -8 )
              {
                v23 = v21[5];
                if ( (_QWORD *)v23 != v21 + 7 )
                  _libc_free(v23);
              }
              v21 += 9;
            }
            while ( v22 != v21 );
          }
          j___libc_free_0(v37);
          sub_15CE080(&v33);
        }
        else
        {
          sub_15D5DA0(a1, a2);
        }
      }
LABEL_18:
      sub_15D6090(a1, a2);
    }
  }
}
