// Function: sub_103A310
// Address: 0x103a310
//
void __fastcall sub_103A310(__int64 a1, char a2, _QWORD *a3, __int64 a4, char **a5)
{
  _QWORD *v5; // r12
  _QWORD *v6; // r14
  _BYTE *v7; // rbx
  _QWORD *v8; // r13
  _QWORD *v9; // rax
  unsigned __int64 v10; // r15
  _QWORD *v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int8 *v14; // r15
  unsigned __int8 v15; // di
  __int64 v16; // rax
  __int64 v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _BOOL8 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v24; // [rsp+8h] [rbp-48h]
  _QWORD *v25; // [rsp+10h] [rbp-40h]

  v5 = &a3[a4];
  if ( a3 == v5 )
    BUG();
  v6 = a3;
  if ( *(_QWORD *)a1 )
  {
    **(_BYTE **)a1 |= a2;
    v7 = *(_BYTE **)a1;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = *a3;
    v21 = sub_22077B0(80);
    v7 = (_BYTE *)v21;
    if ( v21 )
    {
      *(_BYTE *)(v21 + 1) = 1;
      v22 = v21 + 40;
      *(_QWORD *)(v22 - 32) = 0;
      *(_BYTE *)(v22 - 40) = a2;
      *(_QWORD *)(v22 - 24) = 0;
      *(_QWORD *)(v22 - 16) = 0;
      *(_DWORD *)v22 = 0;
      *(_QWORD *)(v22 + 8) = 0;
      *((_QWORD *)v7 + 7) = v22;
      *((_QWORD *)v7 + 8) = v22;
      *((_QWORD *)v7 + 9) = 0;
    }
    *(_QWORD *)a1 = v7;
  }
  while ( 1 )
  {
    v8 = v6 + 1;
    v6 = v8;
    if ( v5 == v8 )
      break;
    while ( 1 )
    {
      v9 = (_QWORD *)*((_QWORD *)v7 + 6);
      v10 = *v8;
      v11 = v7 + 40;
      if ( v9 )
      {
        do
        {
          while ( 1 )
          {
            v12 = v9[2];
            v13 = v9[3];
            if ( v10 <= v9[4] )
              break;
            v9 = (_QWORD *)v9[3];
            if ( !v13 )
              goto LABEL_10;
          }
          v11 = v9;
          v9 = (_QWORD *)v9[2];
        }
        while ( v12 );
LABEL_10:
        if ( v7 + 40 != (_BYTE *)v11 && v10 >= v11[4] )
          break;
      }
      v24 = v7 + 40;
      v16 = sub_22077B0(48);
      *(_QWORD *)(v16 + 32) = v10;
      v17 = v16;
      *(_QWORD *)(v16 + 40) = 0;
      v18 = sub_103A210((_QWORD *)v7 + 4, v11, (unsigned __int64 *)(v16 + 32));
      if ( v19 )
      {
        v20 = v18 || v24 == (_BYTE *)v19 || v10 < *(_QWORD *)(v19 + 32);
        sub_220F040(v20, v17, v19, v24);
        ++*((_QWORD *)v7 + 9);
      }
      else
      {
        v25 = v18;
        j_j___libc_free_0(v17, 48);
        v17 = (__int64)v25;
      }
      v7 = (_BYTE *)sub_22077B0(80);
      if ( v7 )
      {
        v7[1] = 1;
        *((_QWORD *)v7 + 1) = 0;
        *v7 = a2;
        *((_QWORD *)v7 + 2) = 0;
        *((_QWORD *)v7 + 3) = 0;
        *((_DWORD *)v7 + 10) = 0;
        *((_QWORD *)v7 + 6) = 0;
        *((_QWORD *)v7 + 7) = v7 + 40;
        *((_QWORD *)v7 + 8) = v7 + 40;
        *((_QWORD *)v7 + 9) = 0;
      }
      ++v8;
      *(_QWORD *)(v17 + 40) = v7;
      v6 = v8;
      if ( v5 == v8 )
        goto LABEL_23;
    }
    v14 = (unsigned __int8 *)v11[5];
    v15 = *v14 | a2;
    *v14 = v15;
    if ( !sub_10394B0(v15) )
      v7[1] = 0;
    v7 = v14;
  }
LABEL_23:
  sub_1038AF0((__int64)(v7 + 8), *((char **)v7 + 2), *a5, a5[1]);
}
