// Function: sub_2D6BC60
// Address: 0x2d6bc60
//
__int64 __fastcall sub_2D6BC60(__int64 a1)
{
  _BYTE *v1; // r8
  __int64 *v2; // rbx
  __int64 v3; // r11
  __int64 v4; // r14
  int v5; // ecx
  __int64 v6; // r15
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  int v9; // eax
  unsigned __int64 v10; // rax
  unsigned int v11; // r9d
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 *v14; // r9
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned __int8 *v18; // rdx
  __int64 *v19; // r13
  int v20; // eax
  unsigned __int64 v21; // rax
  unsigned int v22; // r15d
  __int64 *v24; // rdx
  int v25; // eax
  __int16 v26; // dx
  __int64 v27; // r10
  char v28; // al
  char v29; // dl
  __int64 v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // rax
  unsigned int v33; // ecx
  __int64 v34; // rdi
  int v35; // r10d
  __int64 *v36; // r11
  int v37; // r10d
  unsigned int v38; // ecx
  __int64 v39; // rdi
  unsigned __int64 *v40; // [rsp+0h] [rbp-70h]
  _BYTE *v41; // [rsp+8h] [rbp-68h]
  _BYTE *v42; // [rsp+8h] [rbp-68h]
  int v43; // [rsp+10h] [rbp-60h]
  __int64 *v44; // [rsp+10h] [rbp-60h]
  _BYTE *v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  _QWORD *v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+20h] [rbp-50h] BYREF
  __int64 v49; // [rsp+28h] [rbp-48h]
  __int64 v50; // [rsp+30h] [rbp-40h]
  unsigned int v51; // [rsp+38h] [rbp-38h]

  v1 = (_BYTE *)a1;
  v2 = *(__int64 **)(a1 + 16);
  v3 = *(_QWORD *)(a1 + 40);
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  if ( !v2 )
    goto LABEL_42;
  v4 = 0x100060000000001LL;
  v5 = 0;
  v6 = v3;
  do
  {
    while ( 1 )
    {
      v18 = (unsigned __int8 *)v2[3];
      v19 = v2;
      v2 = (__int64 *)v2[1];
      v20 = *v18;
      if ( (_BYTE)v20 != 84 )
        break;
      v7 = *(_QWORD *)(*((_QWORD *)v18 - 1)
                     + 32LL * *((unsigned int *)v18 + 18)
                     + 8LL * (unsigned int)(((__int64)v19 - *((_QWORD *)v18 - 1)) >> 5));
LABEL_4:
      v8 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 == v7 + 48 )
        goto LABEL_72;
      if ( !v8 )
        BUG();
      v9 = *(unsigned __int8 *)(v8 - 24);
      if ( (unsigned int)(v9 - 30) > 0xA )
LABEL_72:
        BUG();
      v10 = (unsigned int)(v9 - 39);
      if ( (unsigned int)v10 <= 0x38 && _bittest64(&v4, v10) || v6 == v7 )
        goto LABEL_20;
      if ( !v51 )
      {
        ++v48;
        goto LABEL_45;
      }
      v11 = (v51 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v12 = (__int64 *)(v49 + 16LL * v11);
      v13 = *v12;
      if ( *v12 != v7 )
      {
        v43 = 1;
        v24 = 0;
        while ( v13 != -4096 )
        {
          if ( v13 == -8192 && !v24 )
            v24 = v12;
          v11 = (v51 - 1) & (v43 + v11);
          v12 = (__int64 *)(v49 + 16LL * v11);
          v13 = *v12;
          if ( *v12 == v7 )
            goto LABEL_12;
          ++v43;
        }
        if ( !v24 )
          v24 = v12;
        ++v48;
        v25 = v50 + 1;
        if ( 4 * ((int)v50 + 1) >= 3 * v51 )
        {
LABEL_45:
          v45 = v1;
          sub_2D6BA80((__int64)&v48, 2 * v51);
          if ( !v51 )
            goto LABEL_71;
          v1 = v45;
          v33 = (v51 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v25 = v50 + 1;
          v24 = (__int64 *)(v49 + 16LL * v33);
          v34 = *v24;
          if ( *v24 != v7 )
          {
            v35 = 1;
            v36 = 0;
            while ( v34 != -4096 )
            {
              if ( v34 == -8192 && !v36 )
                v36 = v24;
              v33 = (v51 - 1) & (v35 + v33);
              v24 = (__int64 *)(v49 + 16LL * v33);
              v34 = *v24;
              if ( *v24 == v7 )
                goto LABEL_34;
              ++v35;
            }
LABEL_49:
            if ( v36 )
              v24 = v36;
          }
        }
        else if ( v51 - HIDWORD(v50) - v25 <= v51 >> 3 )
        {
          v42 = v1;
          sub_2D6BA80((__int64)&v48, v51);
          if ( !v51 )
          {
LABEL_71:
            LODWORD(v50) = v50 + 1;
            BUG();
          }
          v36 = 0;
          v1 = v42;
          v37 = 1;
          v38 = (v51 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v25 = v50 + 1;
          v24 = (__int64 *)(v49 + 16LL * v38);
          v39 = *v24;
          if ( *v24 != v7 )
          {
            while ( v39 != -4096 )
            {
              if ( !v36 && v39 == -8192 )
                v36 = v24;
              v38 = (v51 - 1) & (v37 + v38);
              v24 = (__int64 *)(v49 + 16LL * v38);
              v39 = *v24;
              if ( *v24 == v7 )
                goto LABEL_34;
              ++v37;
            }
            goto LABEL_49;
          }
        }
LABEL_34:
        LODWORD(v50) = v25;
        if ( *v24 != -4096 )
          --HIDWORD(v50);
        *v24 = v7;
        v14 = v24 + 1;
        v24[1] = 0;
        goto LABEL_37;
      }
LABEL_12:
      v14 = v12 + 1;
      v15 = v12[1];
      if ( v15 )
      {
        if ( !*v19 || (v16 = v19[1], (*(_QWORD *)v19[2] = v16) == 0) )
        {
          *v19 = v15;
LABEL_17:
          v17 = *(_QWORD *)(v15 + 16);
          v19[1] = v17;
          if ( v17 )
            *(_QWORD *)(v17 + 16) = v19 + 1;
          v19[2] = v15 + 16;
          v5 = 1;
          *(_QWORD *)(v15 + 16) = v19;
          goto LABEL_20;
        }
LABEL_15:
        *(_QWORD *)(v16 + 16) = v19[2];
        goto LABEL_16;
      }
LABEL_37:
      v41 = v1;
      v44 = v14;
      v27 = sub_AA5190(v7);
      if ( v27 )
      {
        v28 = v26;
        v29 = HIBYTE(v26);
      }
      else
      {
        v29 = 0;
        v28 = 0;
      }
      v30 = v46;
      v40 = (unsigned __int64 *)v27;
      LOBYTE(v30) = v28;
      v31 = v30;
      BYTE1(v31) = v29;
      v46 = v31;
      v32 = (_QWORD *)sub_B47F80(v41);
      *v44 = (__int64)v32;
      sub_B44150(v32, v7, v40, v46);
      v1 = v41;
      v15 = *v44;
      if ( *v19 )
      {
        v16 = v19[1];
        *(_QWORD *)v19[2] = v16;
        if ( v16 )
          goto LABEL_15;
      }
LABEL_16:
      *v19 = v15;
      v5 = 1;
      if ( v15 )
        goto LABEL_17;
LABEL_20:
      if ( !v2 )
        goto LABEL_25;
    }
    v21 = (unsigned int)(v20 - 39);
    v7 = *((_QWORD *)v18 + 5);
    if ( (unsigned int)v21 > 0x38 || !_bittest64(&v4, v21) )
      goto LABEL_4;
  }
  while ( v2 );
LABEL_25:
  v22 = v5;
  if ( !*((_QWORD *)v1 + 2) )
  {
LABEL_42:
    v47 = v1;
    v22 = 1;
    sub_F54ED0(v1);
    sub_B43D60(v47);
  }
  sub_C7D6A0(v49, 16LL * v51, 8);
  return v22;
}
