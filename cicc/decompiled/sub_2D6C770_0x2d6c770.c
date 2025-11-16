// Function: sub_2D6C770
// Address: 0x2d6c770
//
__int64 __fastcall sub_2D6C770(unsigned __int8 *a1, unsigned __int8 *a2)
{
  unsigned int v2; // r14d
  __int64 (*v4)(); // rax
  __int64 *v5; // rbx
  int v6; // r10d
  __int64 *v7; // rdx
  unsigned int v8; // edi
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // r12
  __int64 v17; // r13
  unsigned int v18; // eax
  int v19; // ecx
  __int64 v20; // r8
  __int16 v21; // dx
  __int64 v22; // r9
  __int64 v23; // rax
  char v24; // dl
  __int64 v25; // rcx
  __int16 v26; // si
  __int64 v27; // rdx
  int v28; // edi
  _QWORD *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r13
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  int v35; // r9d
  unsigned int v36; // r14d
  __int64 *v37; // r8
  __int64 v38; // rsi
  int v39; // r10d
  __int64 *v40; // r9
  unsigned __int64 *v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+10h] [rbp-90h]
  __int64 v44; // [rsp+18h] [rbp-88h]
  __int64 v45; // [rsp+20h] [rbp-80h] BYREF
  __int64 v46; // [rsp+28h] [rbp-78h]
  __int64 v47; // [rsp+30h] [rbp-70h]
  unsigned int v48; // [rsp+38h] [rbp-68h]
  __int64 v49[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v50; // [rsp+60h] [rbp-40h]

  v2 = a2[16];
  if ( (_BYTE)v2 )
    return 0;
  v4 = *(__int64 (**)())(*(_QWORD *)a2 + 24LL);
  if ( v4 != sub_2D56580 && ((unsigned __int8 (__fastcall *)(unsigned __int8 *))v4)(a2) && *a1 == 83 )
    return 0;
  v5 = (__int64 *)*((_QWORD *)a1 + 2);
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  if ( v5 )
  {
    while ( 1 )
    {
      v15 = v5[3];
      v16 = v5;
      v5 = (__int64 *)v5[1];
      if ( *(_BYTE *)v15 == 84 )
        goto LABEL_14;
      v17 = *(_QWORD *)(v15 + 40);
      if ( *((_QWORD *)a1 + 5) == v17 )
        goto LABEL_14;
      if ( !v48 )
        break;
      v6 = 1;
      v7 = 0;
      v8 = (v48 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v9 = (__int64 *)(v46 + 16LL * v8);
      v10 = *v9;
      if ( v17 != *v9 )
      {
        while ( v10 != -4096 )
        {
          if ( v10 == -8192 && !v7 )
            v7 = v9;
          v8 = (v48 - 1) & (v6 + v8);
          v9 = (__int64 *)(v46 + 16LL * v8);
          v10 = *v9;
          if ( v17 == *v9 )
            goto LABEL_6;
          ++v6;
        }
        if ( !v7 )
          v7 = v9;
        ++v45;
        v19 = v47 + 1;
        if ( 4 * ((int)v47 + 1) < 3 * v48 )
        {
          if ( v48 - HIDWORD(v47) - v19 <= v48 >> 3 )
          {
            sub_2D6C590((__int64)&v45, v48);
            if ( !v48 )
            {
LABEL_80:
              LODWORD(v47) = v47 + 1;
              BUG();
            }
            v35 = 1;
            v36 = (v48 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v37 = 0;
            v19 = v47 + 1;
            v7 = (__int64 *)(v46 + 16LL * v36);
            v38 = *v7;
            if ( v17 != *v7 )
            {
              while ( v38 != -4096 )
              {
                if ( !v37 && v38 == -8192 )
                  v37 = v7;
                v36 = (v48 - 1) & (v35 + v36);
                v7 = (__int64 *)(v46 + 16LL * v36);
                v38 = *v7;
                if ( v17 == *v7 )
                  goto LABEL_21;
                ++v35;
              }
              if ( v37 )
                v7 = v37;
            }
          }
          goto LABEL_21;
        }
LABEL_19:
        sub_2D6C590((__int64)&v45, 2 * v48);
        if ( !v48 )
          goto LABEL_80;
        v18 = (v48 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v19 = v47 + 1;
        v7 = (__int64 *)(v46 + 16LL * v18);
        v20 = *v7;
        if ( v17 != *v7 )
        {
          v39 = 1;
          v40 = 0;
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v40 )
              v40 = v7;
            v18 = (v48 - 1) & (v39 + v18);
            v7 = (__int64 *)(v46 + 16LL * v18);
            v20 = *v7;
            if ( v17 == *v7 )
              goto LABEL_21;
            ++v39;
          }
          if ( v40 )
            v7 = v40;
        }
LABEL_21:
        LODWORD(v47) = v19;
        if ( *v7 != -4096 )
          --HIDWORD(v47);
        *v7 = v17;
        v11 = v7 + 1;
        v7[1] = 0;
        goto LABEL_24;
      }
LABEL_6:
      v11 = v9 + 1;
      v12 = v9[1];
      if ( v12 )
      {
        if ( !*v16 || (v13 = v16[1], (*(_QWORD *)v16[2] = v13) == 0) )
        {
          *v16 = v12;
LABEL_11:
          v14 = *(_QWORD *)(v12 + 16);
          v16[1] = v14;
          if ( v14 )
            *(_QWORD *)(v14 + 16) = v16 + 1;
          v16[2] = v12 + 16;
          v2 = 1;
          *(_QWORD *)(v12 + 16) = v16;
          goto LABEL_14;
        }
LABEL_9:
        *(_QWORD *)(v13 + 16) = v16[2];
        goto LABEL_10;
      }
LABEL_24:
      v22 = sub_AA5190(v17);
      if ( v22 )
      {
        LOBYTE(v23) = v21;
        v24 = HIBYTE(v21);
      }
      else
      {
        v24 = 0;
        LOBYTE(v23) = 0;
      }
      v25 = *((_QWORD *)a1 - 4);
      v23 = (unsigned __int8)v23;
      v26 = *((_WORD *)a1 + 1) & 0x3F;
      v41 = (unsigned __int64 *)v22;
      BYTE1(v23) = v24;
      v27 = *((_QWORD *)a1 - 8);
      v28 = *a1 - 29;
      v42 = v23;
      v50 = 257;
      v29 = (_QWORD *)sub_B52500(v28, v26, v27, v25, (__int64)v49, v22, 0, 0);
      *v11 = (__int64)v29;
      sub_B44150(v29, v17, v41, v42);
      v30 = *v11;
      v49[0] = *((_QWORD *)a1 + 6);
      if ( !v49[0] )
      {
        v31 = v30 + 48;
        if ( (__int64 *)(v30 + 48) == v49 )
          goto LABEL_30;
        v33 = *(_QWORD *)(v30 + 48);
        if ( !v33 )
          goto LABEL_30;
LABEL_42:
        v44 = v30;
        sub_B91220(v31, v33);
        v30 = v44;
        goto LABEL_43;
      }
      v43 = v30;
      sub_B96E90((__int64)v49, v49[0], 1);
      v30 = v43;
      v31 = v43 + 48;
      if ( (__int64 *)(v43 + 48) == v49 )
      {
        if ( v49[0] )
          sub_B91220(v43 + 48, v49[0]);
        goto LABEL_30;
      }
      v33 = *(_QWORD *)(v43 + 48);
      if ( v33 )
        goto LABEL_42;
LABEL_43:
      v34 = (unsigned __int8 *)v49[0];
      *(_QWORD *)(v30 + 48) = v49[0];
      if ( v34 )
        sub_B976B0((__int64)v49, v34, v31);
LABEL_30:
      v12 = *v11;
      if ( *v16 )
      {
        v13 = v16[1];
        *(_QWORD *)v16[2] = v13;
        if ( v13 )
          goto LABEL_9;
      }
LABEL_10:
      *v16 = v12;
      v2 = 1;
      if ( v12 )
        goto LABEL_11;
LABEL_14:
      if ( !v5 )
      {
        if ( *((_QWORD *)a1 + 2) )
          goto LABEL_35;
        goto LABEL_47;
      }
    }
    ++v45;
    goto LABEL_19;
  }
LABEL_47:
  v2 = 1;
  sub_B43D60(a1);
LABEL_35:
  sub_C7D6A0(v46, 16LL * v48, 8);
  return v2;
}
