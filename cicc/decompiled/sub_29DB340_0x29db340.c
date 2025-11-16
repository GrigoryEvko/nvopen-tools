// Function: sub_29DB340
// Address: 0x29db340
//
__int64 __fastcall sub_29DB340(__int64 *a1)
{
  unsigned int v1; // r14d
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // rsi
  __int64 v6; // rdx
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // r12
  unsigned __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  int v19; // edx
  __int64 v20; // r12
  __int64 v21; // rax
  unsigned int v22; // r15d
  int v23; // r13d
  __int64 *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v29; // rax
  char v30; // dl
  __int64 v31; // rax
  __int64 v32; // r9
  __int64 v33; // rdx
  unsigned __int64 v34; // r8
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rdx
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // rdi
  __int64 v41; // [rsp+20h] [rbp-200h]
  __int64 v42; // [rsp+20h] [rbp-200h]
  _QWORD *v44; // [rsp+30h] [rbp-1F0h] BYREF
  unsigned int v45; // [rsp+38h] [rbp-1E8h]
  unsigned int v46; // [rsp+3Ch] [rbp-1E4h]
  _QWORD v47[8]; // [rsp+40h] [rbp-1E0h] BYREF
  _QWORD *v48; // [rsp+80h] [rbp-1A0h] BYREF
  unsigned int v49; // [rsp+88h] [rbp-198h]
  unsigned int v50; // [rsp+8Ch] [rbp-194h]
  _QWORD v51[8]; // [rsp+90h] [rbp-190h] BYREF
  __int64 v52; // [rsp+D0h] [rbp-150h] BYREF
  __int64 *v53; // [rsp+D8h] [rbp-148h]
  unsigned int v54; // [rsp+E0h] [rbp-140h]
  unsigned int v55; // [rsp+E4h] [rbp-13Ch]
  int v56; // [rsp+E8h] [rbp-138h]
  char v57; // [rsp+ECh] [rbp-134h]
  __int64 v58; // [rsp+F0h] [rbp-130h] BYREF

  sub_29D89A0((__int64)(a1 + 2));
  sub_29D89A0((__int64)(a1 + 6));
  v1 = sub_29DB0A0(a1);
  if ( v1 )
    return v1;
  v46 = 8;
  v44 = v47;
  v48 = v51;
  v53 = &v58;
  v2 = *a1;
  v50 = 8;
  v3 = *(_QWORD *)(v2 + 80);
  v57 = 1;
  v54 = 32;
  v56 = 0;
  if ( v3 )
  {
    v3 -= 24;
    v4 = v3;
  }
  else
  {
    v4 = 0;
  }
  v47[0] = v4;
  v5 = v51;
  v6 = a1[1];
  v45 = 1;
  v7 = v47;
  v8 = *(_QWORD *)(v6 + 80);
  v49 = 1;
  v55 = 1;
  v9 = v8 - 24;
  v52 = 1;
  if ( !v8 )
    v9 = 0;
  v58 = v3;
  v10 = 1;
  v51[0] = v9;
  v11 = 1;
  while ( 1 )
  {
    v12 = v7[v10 - 1];
    v45 = v10 - 1;
    v13 = v5[v11 - 1];
    v49 = v11 - 1;
    v1 = sub_29DA390((__int64)a1, v12, v13);
    if ( v1 )
      break;
    v1 = sub_29DAF30(a1, v12, v13);
    if ( v1 )
      break;
    v14 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v14 == v12 + 48 )
    {
      v16 = 0;
    }
    else
    {
      if ( !v14 )
        BUG();
      v15 = *(unsigned __int8 *)(v14 - 24);
      v16 = 0;
      v17 = v14 - 24;
      if ( (unsigned int)(v15 - 30) < 0xB )
        v16 = v17;
    }
    v18 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v18 == v13 + 48 )
    {
      v20 = 0;
    }
    else
    {
      if ( !v18 )
        BUG();
      v19 = *(unsigned __int8 *)(v18 - 24);
      v20 = 0;
      v21 = v18 - 24;
      if ( (unsigned int)(v19 - 30) < 0xB )
        v20 = v21;
    }
    v22 = 0;
    v23 = sub_B46E30(v16);
    if ( v23 )
    {
      while ( 1 )
      {
        v26 = sub_B46EC0(v16, v22);
        if ( v57 )
        {
          v29 = v53;
          v25 = v55;
          v24 = &v53[v55];
          if ( v53 != v24 )
          {
            while ( v26 != *v29 )
            {
              if ( v24 == ++v29 )
                goto LABEL_33;
            }
            goto LABEL_23;
          }
LABEL_33:
          if ( v55 < v54 )
          {
            ++v55;
            *v24 = v26;
            ++v52;
            goto LABEL_27;
          }
        }
        sub_C8CC70((__int64)&v52, v26, (__int64)v24, v25, v27, v28);
        if ( v30 )
        {
LABEL_27:
          v31 = sub_B46EC0(v16, v22);
          v33 = v45;
          v34 = v45 + 1LL;
          if ( v34 > v46 )
          {
            v42 = v31;
            sub_C8D5F0((__int64)&v44, v47, v45 + 1LL, 8u, v34, v32);
            v33 = v45;
            v31 = v42;
          }
          v44[v33] = v31;
          ++v45;
          v35 = sub_B46EC0(v20, v22);
          v37 = v49;
          v38 = v49 + 1LL;
          if ( v38 > v50 )
          {
            v41 = v35;
            sub_C8D5F0((__int64)&v48, v51, v49 + 1LL, 8u, v38, v36);
            v37 = v49;
            v35 = v41;
          }
          ++v22;
          v48[v37] = v35;
          ++v49;
          if ( v23 == v22 )
            break;
        }
        else
        {
LABEL_23:
          if ( v23 == ++v22 )
            break;
        }
      }
    }
    v10 = v45;
    if ( !v45 )
      break;
    v5 = v48;
    v11 = v49;
    v7 = v44;
  }
  if ( v57 )
  {
    v39 = (unsigned __int64)v48;
    if ( v48 == v51 )
      goto LABEL_38;
    goto LABEL_37;
  }
  _libc_free((unsigned __int64)v53);
  v39 = (unsigned __int64)v48;
  if ( v48 != v51 )
LABEL_37:
    _libc_free(v39);
LABEL_38:
  if ( v44 != v47 )
    _libc_free((unsigned __int64)v44);
  return v1;
}
