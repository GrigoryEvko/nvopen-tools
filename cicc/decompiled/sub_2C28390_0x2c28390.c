// Function: sub_2C28390
// Address: 0x2c28390
//
void __fastcall sub_2C28390(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r14
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rdi
  _QWORD *v13; // rax
  char v14; // dl
  unsigned __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 *v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int64 *v26; // rdi
  __int64 v27; // [rsp+8h] [rbp-138h]
  __int64 v28; // [rsp+10h] [rbp-130h]
  __int64 v29; // [rsp+28h] [rbp-118h] BYREF
  _QWORD v30[2]; // [rsp+30h] [rbp-110h] BYREF
  __int64 v31[2]; // [rsp+40h] [rbp-100h] BYREF
  void *v32; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v33; // [rsp+70h] [rbp-D0h]
  _QWORD *v34; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+88h] [rbp-B8h]
  _QWORD v36[22]; // [rsp+90h] [rbp-B0h] BYREF

  v6 = 0x1FE0780820LL;
  v8 = v36;
  v36[0] = a2;
  v34 = v36;
  v35 = 0x1000000001LL;
  v9 = 1;
  do
  {
    v10 = v9;
    v11 = v8[v9 - 1];
    v12 = *a1;
    LODWORD(v35) = v9 - 1;
    if ( !*(_BYTE *)(v12 + 28) )
      goto LABEL_11;
    v13 = *(_QWORD **)(v12 + 8);
    a4 = *(unsigned int *)(v12 + 20);
    v10 = (__int64)&v13[a4];
    if ( v13 == (_QWORD *)v10 )
    {
LABEL_21:
      if ( (unsigned int)a4 >= *(_DWORD *)(v12 + 16) )
      {
LABEL_11:
        sub_C8CC70(v12, v11, v10, a4, a5, a6);
        if ( !v14 )
          goto LABEL_7;
        v15 = *(unsigned __int8 *)(v11 + 8);
        if ( (unsigned __int8)v15 > 0x24u )
          goto LABEL_13;
      }
      else
      {
        a4 = (unsigned int)(a4 + 1);
        *(_DWORD *)(v12 + 20) = a4;
        *(_QWORD *)v10 = v11;
        ++*(_QWORD *)v12;
        v15 = *(unsigned __int8 *)(v11 + 8);
        if ( (unsigned __int8)v15 > 0x24u )
          goto LABEL_13;
      }
      if ( _bittest64(&v6, v15) )
        goto LABEL_7;
      if ( (unsigned __int8)v15 > 0x17u )
        goto LABEL_13;
      v20 = 8860176;
      if ( !_bittest64(&v20, v15) )
        goto LABEL_13;
      if ( (_BYTE)v15 != 23 )
      {
        if ( (_BYTE)v15 == 9 )
        {
          if ( **(_BYTE **)(v11 + 136) != 58 )
            goto LABEL_28;
          goto LABEL_31;
        }
        if ( (_BYTE)v15 != 16 )
        {
          if ( (_BYTE)v15 != 4 || *(_BYTE *)(v11 + 160) != 29 )
            goto LABEL_28;
          goto LABEL_31;
        }
      }
      if ( *(_DWORD *)(v11 + 160) != 29 )
      {
LABEL_28:
        sub_2C26CD0(v11);
LABEL_13:
        a5 = *(_QWORD *)(v11 + 48);
        v16 = (__int64 *)(a5 + 8LL * *(unsigned int *)(v11 + 56));
        if ( (__int64 *)a5 != v16 )
        {
          v17 = *(__int64 **)(v11 + 48);
          do
          {
            v18 = sub_2BF0490(*v17);
            if ( v18 )
            {
              v19 = (unsigned int)v35;
              a6 = (unsigned int)v35 + 1LL;
              if ( a6 > HIDWORD(v35) )
              {
                v28 = v18;
                sub_C8D5F0((__int64)&v34, v36, (unsigned int)v35 + 1LL, 8u, a5, a6);
                v19 = (unsigned int)v35;
                v18 = v28;
              }
              a4 = (__int64)v34;
              v34[v19] = v18;
              LODWORD(v35) = v35 + 1;
            }
            ++v17;
          }
          while ( v16 != v17 );
        }
        goto LABEL_7;
      }
LABEL_31:
      v21 = *(__int64 **)(v11 + 48);
      v22 = *v21;
      if ( *v21 )
      {
        v23 = v21[1];
        if ( v23 )
        {
          if ( (*(_BYTE *)(v11 + 156) & 1) != 0 )
          {
            v24 = *(_QWORD *)(v11 + 80);
            v33 = 257;
            v30[0] = v24;
            v30[1] = v11 + 24;
            v29 = *(_QWORD *)(v11 + 88);
            if ( v29 )
            {
              v27 = v23;
              sub_2C25AB0(&v29);
              v23 = v27;
            }
            v31[1] = v23;
            v31[0] = v22;
            v25 = sub_2C28020(v30, 13, v31, 2, 0, &v29, &v32);
            sub_9C6650(&v29);
            *(_QWORD *)(v25 + 136) = *(_QWORD *)(v11 + 136);
            sub_2BF1250(v11 + 96, v25 + 96);
            v26 = (__int64 *)v11;
            v11 = v25;
            sub_2C19E60(v26);
            goto LABEL_13;
          }
        }
      }
      goto LABEL_28;
    }
    while ( v11 != *v13 )
    {
      if ( (_QWORD *)v10 == ++v13 )
        goto LABEL_21;
    }
LABEL_7:
    v9 = v35;
    v8 = v34;
  }
  while ( (_DWORD)v35 );
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
}
