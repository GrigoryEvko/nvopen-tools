// Function: sub_AC5700
// Address: 0xac5700
//
unsigned __int8 *__fastcall sub_AC5700(__int64 a1)
{
  __int64 v1; // r10
  __int64 v2; // rax
  __int64 *v3; // r13
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 *v6; // rax
  int v7; // edx
  __int64 *v8; // rsi
  __int64 v9; // rdi
  unsigned int v10; // r12d
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned __int8 *v13; // rax
  __int64 v14; // r10
  unsigned __int8 *v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r12
  __int64 v22; // rax
  unsigned __int8 v23; // al
  __int64 v24; // r12
  _QWORD *v25; // r14
  __int64 v26; // r13
  __int64 v27; // r10
  __int64 v28; // rcx
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // r14
  __int64 v32; // rax
  int v33; // r9d
  __int64 v34; // r13
  __int64 v35; // r14
  int v36; // edx
  int v37; // r12d
  __int64 v38; // rax
  _QWORD *v39; // rdi
  _QWORD *v40; // rax
  __int64 v41; // rsi
  int v42; // edx
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // [rsp-8h] [rbp-C8h]
  int v46; // [rsp+Ch] [rbp-B4h]
  unsigned int v47; // [rsp+10h] [rbp-B0h]
  int v48; // [rsp+10h] [rbp-B0h]
  __int64 v49; // [rsp+18h] [rbp-A8h]
  __int64 v50; // [rsp+18h] [rbp-A8h]
  int v51; // [rsp+18h] [rbp-A8h]
  int v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+28h] [rbp-98h]
  _BYTE v54[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v55; // [rsp+50h] [rbp-70h]
  __int64 *v56; // [rsp+60h] [rbp-60h] BYREF
  __int64 v57; // [rsp+68h] [rbp-58h]
  _BYTE v58[80]; // [rsp+70h] [rbp-50h] BYREF

  v1 = a1;
  v2 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v3 = *(__int64 **)(a1 - 8);
    v4 = &v3[(unsigned __int64)v2 / 8];
  }
  else
  {
    v4 = (__int64 *)a1;
    v3 = (__int64 *)(a1 - v2);
  }
  v56 = (__int64 *)v58;
  v5 = v2 >> 5;
  v57 = 0x400000000LL;
  if ( (unsigned __int64)v2 > 0x80 )
  {
    sub_C8D5F0(&v56, v58, v2 >> 5, 8);
    v8 = v56;
    v7 = v57;
    v1 = a1;
    v6 = &v56[(unsigned int)v57];
  }
  else
  {
    v6 = (__int64 *)v58;
    v7 = 0;
    v8 = (__int64 *)v58;
  }
  if ( v3 != v4 )
  {
    do
    {
      if ( v6 )
        *v6 = *v3;
      v3 += 4;
      ++v6;
    }
    while ( v3 != v4 );
    v8 = v56;
    v7 = v57;
  }
  v9 = *(unsigned __int16 *)(v1 + 2);
  v10 = v7 + v5;
  LODWORD(v57) = v10;
  switch ( (__int16)v9 )
  {
    case '"':
      v23 = *(_BYTE *)(v1 + 1);
      v55 = 257;
      v24 = v10 - 1LL;
      v25 = v8 + 1;
      v47 = v23 >> 1;
      v50 = *v8;
      v46 = v24 + 1;
      v26 = sub_BB5290(v1, v8, 257);
      v15 = (unsigned __int8 *)sub_BD2C40(88, (unsigned int)(v24 + 1));
      if ( v15 )
      {
        v27 = *(_QWORD *)(v50 + 8);
        v28 = v46 & 0x7FFFFFF;
        if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 > 1 )
        {
          v39 = &v25[v24];
          if ( v25 != v39 )
          {
            v40 = v8 + 1;
            while ( 1 )
            {
              v41 = *(_QWORD *)(*v40 + 8LL);
              v42 = *(unsigned __int8 *)(v41 + 8);
              if ( v42 == 17 )
              {
                v43 = 0;
                goto LABEL_40;
              }
              if ( v42 == 18 )
                break;
              if ( v39 == ++v40 )
                goto LABEL_27;
            }
            v43 = 1;
LABEL_40:
            BYTE4(v53) = v43;
            LODWORD(v53) = *(_DWORD *)(v41 + 32);
            v44 = sub_BCE1B0(v27, v53);
            v28 = v46 & 0x7FFFFFF;
            v27 = v44;
          }
        }
LABEL_27:
        sub_B44260(v15, v27, 34, v28, 0, 0);
        *((_QWORD *)v15 + 9) = v26;
        *((_QWORD *)v15 + 10) = sub_B4DC50(v26, v25, v24);
        sub_B4D9A0(v15, v50, v25, v24, v54);
      }
      v12 = v47;
      sub_B4DDE0(v15, v47);
      break;
    case '&':
    case '/':
    case '0':
    case '1':
    case '2':
      v19 = *(_QWORD *)(v1 + 8);
      v55 = 257;
      v12 = *v8;
      v15 = (unsigned __int8 *)sub_B51D30(v9, v12, v19, v54, 0, 0);
      break;
    case '=':
      v55 = 257;
      v20 = v8[1];
      v21 = *v8;
      v12 = 2;
      v22 = sub_BD2C40(72, 2);
      v15 = (unsigned __int8 *)v22;
      if ( v22 )
      {
        v12 = v21;
        sub_B4DE80(v22, v21, v20, v54, 0, 0);
      }
      break;
    case '>':
      v55 = 257;
      v29 = v8[2];
      v30 = v8[1];
      v31 = *v8;
      v12 = 3;
      v48 = v29;
      v51 = v30;
      v32 = sub_BD2C40(72, 3);
      v15 = (unsigned __int8 *)v32;
      if ( v32 )
      {
        v12 = v31;
        sub_B4DFA0(v32, v31, v51, v48, (unsigned int)v54, v33, 0, 0);
      }
      break;
    case '?':
      v34 = *v8;
      v35 = v8[1];
      v52 = sub_AC35F0(v1);
      v37 = v36;
      v55 = 257;
      v12 = unk_3F1FE60;
      v38 = sub_BD2C40(112, unk_3F1FE60);
      v15 = (unsigned __int8 *)v38;
      if ( v38 )
      {
        sub_B4E9E0(v38, v34, v35, v52, v37, (unsigned int)v54, 0, 0);
        v12 = v45;
      }
      break;
    default:
      v49 = v1;
      v55 = 257;
      v11 = v8[1];
      v12 = *v8;
      v13 = (unsigned __int8 *)sub_B504D0(v9, v12, v11, v54, 0, 0);
      v14 = v49;
      v15 = v13;
      v16 = *v13;
      if ( (unsigned __int8)v16 <= 0x36u )
      {
        v17 = 0x40540000000000LL;
        if ( _bittest64(&v17, v16) )
        {
          sub_B447F0(v15, (*(_BYTE *)(v49 + 1) & 2) != 0);
          v12 = (*(_BYTE *)(v49 + 1) & 4) != 0;
          sub_B44850(v15, v12);
          LODWORD(v16) = *v15;
          v14 = v49;
        }
      }
      if ( (unsigned __int8)(v16 - 55) <= 1u || (unsigned int)(v16 - 48) <= 1 )
      {
        v12 = (*(_BYTE *)(v14 + 1) & 2) != 0;
        sub_B448B0(v15, v12);
      }
      break;
  }
  if ( v56 != (__int64 *)v58 )
    _libc_free(v56, v12);
  return v15;
}
