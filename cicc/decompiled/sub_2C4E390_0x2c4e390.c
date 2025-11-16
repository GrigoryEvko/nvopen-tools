// Function: sub_2C4E390
// Address: 0x2c4e390
//
unsigned int *__fastcall sub_2C4E390(unsigned int *a1, __int64 a2, unsigned int *a3, __int64 **a4, _BYTE **a5)
{
  __int64 v5; // rsi
  unsigned int *v6; // r9
  __int64 v7; // rbx
  _BYTE *v9; // r14
  __int64 v10; // r13
  __int64 v11; // r15
  unsigned int *v12; // rdx
  __int64 v13; // r12
  unsigned __int8 *v14; // rax
  __int64 v15; // rsi
  char v16; // di
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v20; // rcx
  _QWORD *v21; // r11
  _QWORD *v22; // rcx
  __int64 v23; // rdi
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  __int64 *v26; // rax
  __int64 *v27; // rax
  _BYTE **v28; // [rsp+0h] [rbp-60h]
  _BYTE **v29; // [rsp+0h] [rbp-60h]
  unsigned int *v30; // [rsp+8h] [rbp-58h]
  unsigned int *v31; // [rsp+8h] [rbp-58h]
  unsigned int *v32; // [rsp+10h] [rbp-50h]
  unsigned int *v33; // [rsp+10h] [rbp-50h]
  _QWORD *v34; // [rsp+18h] [rbp-48h]
  unsigned int *v35; // [rsp+18h] [rbp-48h]
  unsigned int *v36; // [rsp+18h] [rbp-48h]
  __int64 *v38; // [rsp+28h] [rbp-38h]

  v5 = a2 - (_QWORD)a1;
  v6 = a1;
  v7 = v5 >> 3;
  if ( v5 > 0 )
  {
    while ( 1 )
    {
      v9 = *a5;
      v10 = *a3;
      v11 = v7 >> 1;
      v12 = &v6[2 * (v7 >> 1)];
      v13 = *v12;
      if ( **a5 != 92 )
        goto LABEL_11;
      v38 = *a4;
      v14 = (unsigned __int8 *)*((_QWORD *)v9 - 4);
      if ( (unsigned int)*v14 - 12 <= 1 )
      {
        v15 = *((_QWORD *)v9 - 8);
        v16 = *(_BYTE *)v15;
        if ( *(_BYTE *)v15 == 92 )
        {
          v20 = *v38;
          if ( !*(_BYTE *)(*v38 + 28) )
          {
            v28 = a5;
            v30 = a3;
            v32 = v6;
            v35 = &v6[2 * (v7 >> 1)];
            v26 = sub_C8CA60(*v38, v15);
            v12 = v35;
            v6 = v32;
            a3 = v30;
            a5 = v28;
            v16 = *v9;
            v38 = *a4;
            if ( v26 )
              goto LABEL_19;
LABEL_29:
            v17 = *((_QWORD *)v9 + 9);
            LODWORD(v10) = *(_DWORD *)(v17 + 4 * v10);
LABEL_20:
            if ( v16 != 92 )
              goto LABEL_11;
            v14 = (unsigned __int8 *)*((_QWORD *)v9 - 4);
            goto LABEL_8;
          }
          v21 = *(_QWORD **)(v20 + 8);
          v34 = &v21[*(unsigned int *)(v20 + 20)];
          if ( v21 != v34 )
          {
            v22 = *(_QWORD **)(v20 + 8);
            while ( v15 != *v22 )
            {
              if ( v34 == ++v22 )
                goto LABEL_29;
            }
LABEL_19:
            v17 = *((_QWORD *)v9 + 9);
            LODWORD(v10) = *(_DWORD *)(*(_QWORD *)(v15 + 72) + 4LL * *(unsigned int *)(v17 + 4 * v10));
            goto LABEL_20;
          }
        }
      }
      v17 = *((_QWORD *)v9 + 9);
      LODWORD(v10) = *(_DWORD *)(v17 + 4 * v10);
LABEL_8:
      if ( (unsigned int)*v14 - 12 <= 1 )
      {
        v18 = *((_QWORD *)v9 - 8);
        if ( *(_BYTE *)v18 == 92 )
        {
          v23 = *v38;
          if ( *(_BYTE *)(*v38 + 28) )
          {
            v24 = *(_QWORD **)(v23 + 8);
            v25 = &v24[*(unsigned int *)(v23 + 20)];
            if ( v24 != v25 )
            {
              while ( v18 != *v24 )
              {
                if ( v25 == ++v24 )
                  goto LABEL_10;
              }
LABEL_27:
              LODWORD(v13) = *(_DWORD *)(*(_QWORD *)(v18 + 72) + 4LL * *(unsigned int *)(v17 + 4 * v13));
              goto LABEL_11;
            }
          }
          else
          {
            v29 = a5;
            v31 = a3;
            v33 = v6;
            v36 = v12;
            v27 = sub_C8CA60(v23, v18);
            v17 = *((_QWORD *)v9 + 9);
            v12 = v36;
            v6 = v33;
            a3 = v31;
            a5 = v29;
            if ( v27 )
              goto LABEL_27;
          }
        }
      }
LABEL_10:
      LODWORD(v13) = *(_DWORD *)(v17 + 4 * v13);
LABEL_11:
      if ( (int)v13 <= (int)v10 )
      {
        v6 = v12 + 2;
        v7 = v7 - v11 - 1;
        if ( v7 <= 0 )
          return v6;
      }
      else
      {
        v7 >>= 1;
        if ( v11 <= 0 )
          return v6;
      }
    }
  }
  return v6;
}
