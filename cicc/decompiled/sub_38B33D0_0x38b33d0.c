// Function: sub_38B33D0
// Address: 0x38b33d0
//
__int64 __fastcall sub_38B33D0(__int64 a1, _QWORD *a2)
{
  _QWORD *v5; // rax
  unsigned __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // r10
  _QWORD *v11; // rsi
  unsigned __int64 *v12; // r15
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  char v15; // di
  unsigned __int64 v16; // r15
  __int64 v17; // rcx
  char *v18; // rsi
  int *v19; // rdi
  _DWORD *v20; // rax
  _DWORD *v21; // rcx
  _DWORD *v22; // rdx
  _DWORD *v23; // rax
  _DWORD *v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rax
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // [rsp+0h] [rbp-F0h]
  _QWORD *v34; // [rsp+8h] [rbp-E8h]
  _QWORD *v35; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v36; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v37; // [rsp+38h] [rbp-B8h] BYREF
  unsigned __int64 v38; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v39; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v40; // [rsp+50h] [rbp-A0h]
  int v41; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 v42[2]; // [rsp+68h] [rbp-88h] BYREF
  _BYTE v43[16]; // [rsp+78h] [rbp-78h] BYREF
  char v44; // [rsp+88h] [rbp-68h] BYREF
  int v45; // [rsp+90h] [rbp-60h] BYREF
  int *v46; // [rsp+98h] [rbp-58h]
  int *v47; // [rsp+A0h] [rbp-50h]
  int *v48; // [rsp+A8h] [rbp-48h]
  __int64 v49; // [rsp+B0h] [rbp-40h]

  if ( !(unsigned __int8)sub_388AF10(a1, 354, "expected 'wpdResolutions' here")
    && !(unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    && !(unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
  {
    while ( 1 )
    {
      v43[0] = 0;
      v42[1] = 0;
      v41 = 0;
      v42[0] = (unsigned __int64)v43;
      v45 = 0;
      v46 = 0;
      v47 = &v45;
      v48 = &v45;
      v49 = 0;
      if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
        || (unsigned __int8)sub_388AF10(a1, 338, "expected 'offset' here")
        || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
        || (unsigned __int8)sub_388BD80(a1, (__int64 *)&v37)
        || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
        || (unsigned __int8)sub_38B31F0(a1, (__int64)&v41)
        || (unsigned __int8)sub_388AF10(a1, 13, "expected ')' here") )
      {
        v30 = (unsigned __int64)v46;
        if ( v46 )
        {
          do
          {
            v31 = v30;
            sub_38885E0(*(_QWORD **)(v30 + 24));
            v32 = *(_QWORD *)(v30 + 32);
            v30 = *(_QWORD *)(v30 + 16);
            if ( v32 )
              j_j___libc_free_0(v32);
            j_j___libc_free_0(v31);
          }
          while ( v30 );
        }
        if ( (_BYTE *)v42[0] != v43 )
          j_j___libc_free_0(v42[0]);
        return 1;
      }
      v5 = (_QWORD *)a2[2];
      if ( !v5 )
        break;
      v6 = (unsigned __int64)(a2 + 1);
      do
      {
        while ( 1 )
        {
          v7 = v5[2];
          v8 = v5[3];
          if ( v5[4] >= v37 )
            break;
          v5 = (_QWORD *)v5[3];
          if ( !v8 )
            goto LABEL_17;
        }
        v6 = (unsigned __int64)v5;
        v5 = (_QWORD *)v5[2];
      }
      while ( v7 );
LABEL_17:
      if ( a2 + 1 == (_QWORD *)v6 || v37 < *(_QWORD *)(v6 + 32) )
        goto LABEL_19;
LABEL_25:
      *(_DWORD *)(v6 + 40) = v41;
      sub_2240AE0((unsigned __int64 *)(v6 + 48), v42);
      if ( (char *)(v6 + 80) == &v44 )
        goto LABEL_38;
      v16 = *(_QWORD *)(v6 + 96);
      v17 = *(_QWORD *)(v6 + 112);
      v40 = v6 + 80;
      v18 = (char *)(v6 + 88);
      v38 = v16;
      v39 = v17;
      if ( v16 )
      {
        *(_QWORD *)(v16 + 8) = 0;
        if ( *(_QWORD *)(v17 + 16) )
          v39 = *(_QWORD *)(v17 + 16);
        v19 = v46;
        *(_QWORD *)(v6 + 96) = 0;
        *(_QWORD *)(v6 + 104) = v18;
        *(_QWORD *)(v6 + 112) = v18;
        *(_QWORD *)(v6 + 120) = 0;
        if ( v19 )
        {
LABEL_30:
          v20 = sub_3887B50(v19, v18, &v38);
          v21 = v20;
          do
          {
            v22 = v20;
            v20 = (_DWORD *)*((_QWORD *)v20 + 2);
          }
          while ( v20 );
          *(_QWORD *)(v6 + 104) = v22;
          v23 = v21;
          do
          {
            v24 = v23;
            v23 = (_DWORD *)*((_QWORD *)v23 + 3);
          }
          while ( v23 );
          v25 = v49;
          v16 = v38;
          *(_QWORD *)(v6 + 112) = v24;
          *(_QWORD *)(v6 + 96) = v21;
          *(_QWORD *)(v6 + 120) = v25;
          if ( v16 )
            goto LABEL_35;
        }
        else
        {
          do
          {
LABEL_35:
            v26 = v16;
            sub_38885E0(*(_QWORD **)(v16 + 24));
            v27 = *(_QWORD *)(v16 + 32);
            v16 = *(_QWORD *)(v16 + 16);
            if ( v27 )
              j_j___libc_free_0(v27);
            j_j___libc_free_0(v26);
          }
          while ( v16 );
        }
LABEL_38:
        v19 = v46;
        goto LABEL_39;
      }
      v19 = v46;
      *(_QWORD *)(v6 + 104) = v18;
      v39 = 0;
      *(_QWORD *)(v6 + 112) = v18;
      *(_QWORD *)(v6 + 120) = 0;
      if ( v19 )
        goto LABEL_30;
LABEL_39:
      sub_38885E0(v19);
      if ( (_BYTE *)v42[0] != v43 )
        j_j___libc_free_0(v42[0]);
      if ( *(_DWORD *)(a1 + 64) != 4 )
        return sub_388AF10(a1, 13, "expected ')' here");
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    }
    v6 = (unsigned __int64)(a2 + 1);
LABEL_19:
    v34 = a2 + 1;
    v9 = (unsigned __int64 *)sub_22077B0(0x80u);
    v10 = v37;
    v11 = (_QWORD *)v6;
    v6 = (unsigned __int64)v9;
    v9[4] = v37;
    v12 = v9 + 8;
    memset(v9 + 5, 0, 0x58u);
    v9[6] = (unsigned __int64)(v9 + 8);
    v9[13] = (unsigned __int64)(v9 + 11);
    v9[14] = (unsigned __int64)(v9 + 11);
    v33 = v10;
    v13 = sub_14F7B80(a2, v11, v9 + 4);
    if ( v14 )
    {
      v15 = v34 == v14 || v13 || v33 < v14[4];
      sub_220F040(v15, v6, v14, v34);
      ++a2[5];
    }
    else
    {
      v35 = v13;
      sub_38885E0(0);
      v28 = *(_QWORD *)(v6 + 48);
      v29 = v35;
      if ( v12 != (unsigned __int64 *)v28 )
      {
        j_j___libc_free_0(v28);
        v29 = v35;
      }
      v36 = (unsigned __int64)v29;
      j_j___libc_free_0(v6);
      v6 = v36;
    }
    goto LABEL_25;
  }
  return 1;
}
