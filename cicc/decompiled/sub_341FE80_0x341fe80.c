// Function: sub_341FE80
// Address: 0x341fe80
//
__int64 __fastcall sub_341FE80(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v6; // r12d
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v12; // rax
  __int64 *v13; // r10
  unsigned int *v14; // r12
  unsigned int *v15; // r11
  __int64 v16; // rbx
  __int64 *v17; // rax
  unsigned int *v18; // r12
  unsigned int *v19; // r14
  __int64 v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned int *v26; // [rsp+20h] [rbp-170h]
  __int64 *v27; // [rsp+20h] [rbp-170h]
  __int64 *v28; // [rsp+28h] [rbp-168h]
  __int64 *v29; // [rsp+28h] [rbp-168h]
  unsigned int *v30; // [rsp+28h] [rbp-168h]
  __int64 *v31; // [rsp+28h] [rbp-168h]
  __int64 *v32; // [rsp+30h] [rbp-160h] BYREF
  __int64 v33; // [rsp+38h] [rbp-158h]
  _BYTE v34[128]; // [rsp+40h] [rbp-150h] BYREF
  __int64 v35; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 *v36; // [rsp+C8h] [rbp-C8h]
  __int64 v37; // [rsp+D0h] [rbp-C0h]
  int v38; // [rsp+D8h] [rbp-B8h]
  char v39; // [rsp+DCh] [rbp-B4h]
  char v40; // [rsp+E0h] [rbp-B0h] BYREF

  v6 = 0;
  v36 = (__int64 *)&v40;
  v32 = (__int64 *)v34;
  v35 = 0;
  v37 = 16;
  v38 = 0;
  v39 = 1;
  v33 = 0x1000000000LL;
  if ( !(unsigned __int8)sub_33CF8D0(a3, a2) )
  {
    if ( v39 )
    {
      v12 = v36;
      v8 = HIDWORD(v37);
      v7 = &v36[HIDWORD(v37)];
      if ( v36 != v7 )
      {
        while ( a3 != *v12 )
        {
          if ( v7 == ++v12 )
            goto LABEL_50;
        }
        goto LABEL_12;
      }
LABEL_50:
      if ( HIDWORD(v37) < (unsigned int)v37 )
      {
        v8 = (unsigned int)++HIDWORD(v37);
        *v7 = a3;
        ++v35;
LABEL_12:
        v13 = &v35;
LABEL_13:
        v14 = *(unsigned int **)(a3 + 40);
        v15 = &v14[10 * *(unsigned int *)(a3 + 64)];
        if ( v14 != v15 )
        {
          while ( 1 )
          {
            v16 = *(_QWORD *)v14;
            if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v14 + 48LL) + 16LL * v14[2]) == 1 && a4 || a2 == v16 )
              goto LABEL_22;
            if ( v39 )
            {
              v17 = v36;
              v8 = HIDWORD(v37);
              v7 = &v36[HIDWORD(v37)];
              if ( v36 != v7 )
              {
                while ( v16 != *v17 )
                {
                  if ( v7 == ++v17 )
                    goto LABEL_45;
                }
                goto LABEL_22;
              }
LABEL_45:
              if ( HIDWORD(v37) < (unsigned int)v37 )
              {
                ++HIDWORD(v37);
                *v7 = v16;
                ++v35;
                goto LABEL_36;
              }
            }
            v26 = v15;
            v28 = v13;
            sub_C8CC70((__int64)v13, *(_QWORD *)v14, (__int64)v7, v8, v9, v10);
            v13 = v28;
            v15 = v26;
            if ( !(_BYTE)v7 )
            {
LABEL_22:
              v14 += 10;
              if ( v15 == v14 )
                break;
            }
            else
            {
LABEL_36:
              v22 = (unsigned int)v33;
              v8 = HIDWORD(v33);
              v23 = (unsigned int)v33 + 1LL;
              if ( v23 > HIDWORD(v33) )
              {
                v27 = v13;
                v30 = v15;
                sub_C8D5F0((__int64)&v32, v34, v23, 8u, v9, v10);
                v22 = (unsigned int)v33;
                v13 = v27;
                v15 = v30;
              }
              v7 = v32;
              v14 += 10;
              v32[v22] = v16;
              LODWORD(v33) = v33 + 1;
              if ( v15 == v14 )
                break;
            }
          }
        }
        if ( a3 == a1 || (v18 = *(unsigned int **)(a1 + 40), v19 = &v18[10 * *(unsigned int *)(a1 + 64)], v18 == v19) )
        {
LABEL_34:
          v6 = sub_3285B00(a2, (__int64)v13, (__int64)&v32, 0, 1, v10);
          goto LABEL_2;
        }
        while ( 1 )
        {
          v20 = *(_QWORD *)v18;
          if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v18 + 48LL) + 16LL * v18[2]) == 1 && a4 || a2 == v20 )
            goto LABEL_33;
          if ( v39 )
          {
            v21 = v36;
            v8 = HIDWORD(v37);
            v7 = &v36[HIDWORD(v37)];
            if ( v36 != v7 )
            {
              while ( v20 != *v21 )
              {
                if ( v7 == ++v21 )
                  goto LABEL_47;
              }
              goto LABEL_33;
            }
LABEL_47:
            if ( HIDWORD(v37) < (unsigned int)v37 )
            {
              ++HIDWORD(v37);
              *v7 = v20;
              ++v35;
              goto LABEL_41;
            }
          }
          v29 = v13;
          sub_C8CC70((__int64)v13, *(_QWORD *)v18, (__int64)v7, v8, v9, v10);
          v13 = v29;
          if ( !(_BYTE)v7 )
          {
LABEL_33:
            v18 += 10;
            if ( v19 == v18 )
              goto LABEL_34;
          }
          else
          {
LABEL_41:
            v24 = (unsigned int)v33;
            v8 = HIDWORD(v33);
            v25 = (unsigned int)v33 + 1LL;
            if ( v25 > HIDWORD(v33) )
            {
              v31 = v13;
              sub_C8D5F0((__int64)&v32, v34, v25, 8u, v9, v10);
              v24 = (unsigned int)v33;
              v13 = v31;
            }
            v7 = v32;
            v18 += 10;
            v32[v24] = v20;
            LODWORD(v33) = v33 + 1;
            if ( v19 == v18 )
              goto LABEL_34;
          }
        }
      }
    }
    sub_C8CC70((__int64)&v35, a3, (__int64)v7, v8, v9, v10);
    v13 = &v35;
    goto LABEL_13;
  }
LABEL_2:
  if ( v32 != (__int64 *)v34 )
    _libc_free((unsigned __int64)v32);
  if ( !v39 )
    _libc_free((unsigned __int64)v36);
  return v6;
}
