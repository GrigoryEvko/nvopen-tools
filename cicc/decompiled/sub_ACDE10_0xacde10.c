// Function: sub_ACDE10
// Address: 0xacde10
//
__int64 __fastcall sub_ACDE10(__int64 a1, __int64 a2, int *a3)
{
  int *v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r12
  _QWORD *v8; // r15
  _QWORD *v9; // r13
  _QWORD *v10; // rdi
  _QWORD *v11; // r15
  int v12; // eax
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // r13
  char v17; // al
  _QWORD *v18; // rax
  _QWORD *v19; // r13
  char v20; // al
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // [rsp+10h] [rbp-C0h]
  __int64 v27; // [rsp+38h] [rbp-98h]
  __int64 v28; // [rsp+38h] [rbp-98h]
  __int64 v29[4]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v30; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v31[3]; // [rsp+68h] [rbp-68h] BYREF
  __int64 v32; // [rsp+80h] [rbp-50h] BYREF
  _QWORD v33[9]; // [rsp+88h] [rbp-48h] BYREF

  v3 = (int *)a2;
  *(_QWORD *)(a1 + 16) = 0;
  v27 = sub_C33690();
  v7 = sub_C33340(a1, a2, v4, v5, v6);
  if ( v27 == v7 )
    sub_C3C5A0(&v30, v27, 1);
  else
    sub_C36740(&v30, v27, 1);
  LODWORD(v32) = -1;
  BYTE4(v32) = 1;
  if ( v30 == v7 )
    sub_C3C840(v33, &v30);
  else
    sub_C338E0(v33, &v30);
  sub_91D830(&v30);
  v8 = *(_QWORD **)(a1 + 8);
  v9 = &v8[5 * *(unsigned int *)(a1 + 24)];
  if ( v8 != v9 )
  {
    while ( 1 )
    {
      if ( !v8 )
        goto LABEL_8;
      v10 = v8 + 1;
      *v8 = v32;
      if ( v7 == v33[0] )
      {
        sub_C3C790(v10, v33);
        v8 += 5;
        if ( v9 == v8 )
          break;
      }
      else
      {
        sub_C33EB0(v10, v33);
LABEL_8:
        v8 += 5;
        if ( v9 == v8 )
          break;
      }
    }
  }
  sub_91D830(v33);
  if ( v27 == v7 )
    sub_C3C5A0(&v32, v27, 1);
  else
    sub_C36740(&v32, v27, 1);
  BYTE4(v30) = 1;
  LODWORD(v30) = -1;
  if ( v7 == v32 )
    sub_C3C840(v31, &v32);
  else
    sub_C338E0(v31, &v32);
  sub_91D830(&v32);
  if ( v27 == v7 )
    sub_C3C5A0(v29, v7, 2);
  else
    sub_C36740(v29, v27, 2);
  BYTE4(v32) = 0;
  LODWORD(v32) = -2;
  if ( v7 == v29[0] )
    sub_C3C840(v33, v29);
  else
    sub_C338E0(v33, v29);
  v11 = (_QWORD *)(a2 + 8);
  sub_91D830(v29);
  if ( (int *)a2 != a3 )
  {
    do
    {
      v12 = *((_DWORD *)v11 - 2);
      if ( v12 == (_DWORD)v30 && *((_BYTE *)v11 - 4) == BYTE4(v30) && *v11 == v31[0] )
      {
        if ( v7 == *v11 )
          v17 = sub_C3E590(v11);
        else
          v17 = sub_C33D00(v11);
        if ( v17 )
          goto LABEL_28;
        v12 = *((_DWORD *)v11 - 2);
      }
      if ( v12 != (_DWORD)v32
        || *((_BYTE *)v11 - 4) != BYTE4(v32)
        || *v11 != v33[0]
        || (v7 == *v11 ? (v20 = sub_C3E590(v11)) : (v20 = sub_C33D00(v11)), !v20) )
      {
        sub_AC6AC0(a1, v3, v29);
        v13 = v29[0];
        *(_DWORD *)v29[0] = *((_DWORD *)v11 - 2);
        v14 = (_QWORD *)(v13 + 8);
        *(_BYTE *)(v13 + 4) = *((_BYTE *)v11 - 4);
        if ( v7 == *(_QWORD *)(v13 + 8) )
        {
          if ( v7 != *v11 )
            goto LABEL_33;
          if ( v11 != v14 )
          {
            v21 = *(_QWORD *)(v13 + 16);
            if ( v21 )
            {
              v22 = 24LL * *(_QWORD *)(v21 - 8);
              v23 = v21 + v22;
              if ( v21 != v21 + v22 )
              {
                do
                {
                  v24 = v13;
                  v28 = v23 - 24;
                  sub_91D830((_QWORD *)(v23 - 24));
                  v13 = v24;
                  v23 = v28;
                }
                while ( *(_QWORD *)(v24 + 16) != v28 );
              }
              j_j_j___libc_free_0_0(v23 - 8);
            }
            goto LABEL_65;
          }
        }
        else
        {
          if ( v7 != *v11 )
          {
            sub_C33870(v13 + 8, v11);
            goto LABEL_26;
          }
LABEL_33:
          if ( v11 != v14 )
          {
            sub_91D830((_QWORD *)(v13 + 8));
            if ( v7 != *v11 )
            {
              sub_C338E0(v14, v11);
              goto LABEL_26;
            }
LABEL_65:
            sub_C3C840(v14, v11);
          }
        }
LABEL_26:
        *(_QWORD *)(v29[0] + 32) = v11[3];
        v11[3] = 0;
        ++*(_DWORD *)(a1 + 16);
        v15 = v11[3];
        if ( v15 )
        {
          sub_91D830((_QWORD *)(v15 + 24));
          sub_BD7260(v15);
          sub_BD2DD0(v15);
        }
      }
LABEL_28:
      if ( v7 == *v11 )
      {
        v18 = (_QWORD *)v11[1];
        if ( v18 )
        {
          v19 = &v18[3 * *(v18 - 1)];
          if ( v18 != v19 )
          {
            do
            {
              v19 -= 3;
              sub_91D830(v19);
            }
            while ( (_QWORD *)v11[1] != v19 );
          }
          j_j_j___libc_free_0_0(v19 - 1);
        }
      }
      else
      {
        sub_C338F0(v11);
      }
      v3 += 10;
      v11 += 5;
    }
    while ( a3 != v3 );
  }
  sub_91D830(v33);
  return sub_91D830(v31);
}
