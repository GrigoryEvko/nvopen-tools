// Function: sub_3209920
// Address: 0x3209920
//
void __fastcall sub_3209920(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v10; // rax
  char *v11; // rdi
  __int64 v12; // rbx
  char *v13; // r15
  unsigned __int64 v14; // rcx
  char *v15; // r15
  __int64 *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rbx
  unsigned __int8 v19; // dl
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int8 v22; // dl
  __int64 v23; // rdi
  unsigned __int64 v24; // rdx
  _BYTE *v25; // r10
  size_t v26; // r9
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdx
  char *v32; // rbx
  __int64 v33; // rdi
  __int64 v34; // rdx
  char *v35; // rax
  char *v36; // r9
  char *v37; // r9
  size_t n; // [rsp+0h] [rbp-E0h]
  _BYTE *src; // [rsp+8h] [rbp-D8h]
  char *v40; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *v42; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v43; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int64 v44; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v45; // [rsp+48h] [rbp-98h]
  char v46; // [rsp+4Ch] [rbp-94h]
  unsigned __int64 v47[2]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v48[2]; // [rsp+60h] [rbp-80h] BYREF
  char *v49; // [rsp+70h] [rbp-70h] BYREF
  __int64 v50; // [rsp+78h] [rbp-68h]
  _BYTE v51[96]; // [rsp+80h] [rbp-60h] BYREF

  v49 = v51;
  v50 = 0x600000000LL;
  v6 = a3 + 88 * a4;
  if ( a3 == v6 )
  {
    v11 = v51;
  }
  else
  {
    v7 = a3;
    v10 = 0;
    do
    {
      while ( !*(_WORD *)(*(_QWORD *)v7 + 20LL) )
      {
        v7 += 88;
        if ( v6 == v7 )
          goto LABEL_8;
      }
      a4 = HIDWORD(v50);
      if ( v10 + 1 > (unsigned __int64)HIDWORD(v50) )
      {
        sub_C8D5F0((__int64)&v49, v51, v10 + 1, 8u, a5, a6);
        v10 = (unsigned int)v50;
      }
      *(_QWORD *)&v49[8 * v10] = v7;
      v7 += 88;
      v10 = (unsigned int)(v50 + 1);
      LODWORD(v50) = v50 + 1;
    }
    while ( v6 != v7 );
LABEL_8:
    v11 = v49;
    v12 = 8 * v10;
    v13 = &v49[8 * v10];
    if ( v49 != v13 )
    {
      v40 = v49;
      _BitScanReverse64(&v14, v12 >> 3);
      sub_31F4C40(v49, (__int64 *)&v49[8 * v10], 2LL * (int)(63 - (v14 ^ 0x3F)));
      if ( (unsigned __int64)v12 > 0x80 )
      {
        v32 = v40 + 128;
        sub_31F4060(v40, v40 + 128);
        if ( v13 != v40 + 128 )
        {
          do
          {
            while ( 1 )
            {
              v33 = *(_QWORD *)v32;
              v34 = *((_QWORD *)v32 - 1);
              v35 = v32 - 8;
              a4 = *(unsigned __int16 *)(**(_QWORD **)v32 + 20LL);
              if ( *(_WORD *)(*(_QWORD *)v34 + 20LL) > (unsigned __int16)a4 )
                break;
              v37 = v32;
              v32 += 8;
              *(_QWORD *)v37 = v33;
              if ( v13 == v32 )
                goto LABEL_11;
            }
            do
            {
              *((_QWORD *)v35 + 1) = v34;
              v36 = v35;
              v34 = *((_QWORD *)v35 - 1);
              v35 -= 8;
              a4 = *(unsigned __int16 *)(*(_QWORD *)v34 + 20LL);
            }
            while ( *(_WORD *)(*(_QWORD *)v33 + 20LL) < (unsigned __int16)a4 );
            v32 += 8;
            *(_QWORD *)v36 = v33;
          }
          while ( v13 != v32 );
        }
      }
      else
      {
        sub_31F4060(v40, v13);
      }
LABEL_11:
      v15 = &v49[8 * (unsigned int)v50];
      if ( v49 != v15 )
      {
        v16 = (__int64 *)v49;
        do
        {
          v17 = *v16++;
          sub_3209550(a1, a2, v17, a4, a5);
        }
        while ( v15 != (char *)v16 );
      }
      goto LABEL_14;
    }
    if ( a3 != v6 )
    {
LABEL_14:
      v18 = a3;
      while ( 1 )
      {
        while ( 1 )
        {
          v30 = *(_QWORD *)v18;
          if ( !*(_WORD *)(*(_QWORD *)v18 + 20LL) )
            break;
LABEL_31:
          v18 += 88;
          if ( v6 == v18 )
            goto LABEL_35;
        }
        if ( *(_BYTE *)(v18 + 80) )
          break;
        v31 = v18;
        v18 += 88;
        sub_3209550(a1, a2, v31, a4, a5);
        if ( v6 == v18 )
        {
LABEL_35:
          v11 = v49;
          goto LABEL_36;
        }
      }
      v19 = *(_BYTE *)(v30 - 16);
      v20 = v30 - 16;
      if ( (v19 & 2) != 0 )
        v21 = *(_QWORD *)(v30 - 32);
      else
        v21 = v20 - 8LL * ((v19 >> 2) & 0xF);
      v42 = *(unsigned __int8 **)(v21 + 24);
      v45 = *(_DWORD *)(v18 + 72);
      if ( v45 > 0x40 )
      {
        sub_C43780((__int64)&v44, (const void **)(v18 + 64));
        v30 = *(_QWORD *)v18;
        v20 = *(_QWORD *)v18 - 16LL;
      }
      else
      {
        v44 = *(_QWORD *)(v18 + 64);
      }
      v46 = *(_BYTE *)(v18 + 76);
      v22 = *(_BYTE *)(v30 - 16);
      if ( (v22 & 2) != 0 )
      {
        v23 = *(_QWORD *)(*(_QWORD *)(v30 - 32) + 8LL);
        if ( v23 )
          goto LABEL_21;
      }
      else
      {
        v23 = *(_QWORD *)(v20 - 8LL * ((v22 >> 2) & 0xF) + 8);
        if ( v23 )
        {
LABEL_21:
          v25 = (_BYTE *)sub_B91420(v23);
          v47[0] = (unsigned __int64)v48;
          v26 = v24;
          v27 = v24;
          if ( &v25[v24] && !v25 )
            sub_426248((__int64)"basic_string::_M_construct null not valid");
          v43 = v24;
          if ( v24 <= 0xF )
          {
            if ( v24 == 1 )
            {
              LOBYTE(v48[0]) = *v25;
              goto LABEL_26;
            }
            if ( !v24 )
              goto LABEL_26;
            v29 = v48;
          }
          else
          {
            n = v24;
            src = v25;
            v28 = sub_22409D0((__int64)v47, &v43, 0);
            v25 = src;
            v47[0] = v28;
            v29 = (_QWORD *)v28;
            v26 = n;
            v48[0] = v43;
          }
          memcpy(v29, v25, v26);
          v27 = v43;
LABEL_26:
          v47[1] = v27;
          *(_BYTE *)(v47[0] + v27) = 0;
          sub_32086D0(a1, v42, (__int64)&v44, (__int64)v47, a5);
          if ( (_QWORD *)v47[0] != v48 )
            j_j___libc_free_0(v47[0]);
          if ( v45 > 0x40 && v44 )
            j_j___libc_free_0_0(v44);
          goto LABEL_31;
        }
      }
      v27 = 0;
      v47[0] = (unsigned __int64)v48;
      goto LABEL_26;
    }
  }
LABEL_36:
  if ( v11 != v51 )
    _libc_free((unsigned __int64)v11);
}
