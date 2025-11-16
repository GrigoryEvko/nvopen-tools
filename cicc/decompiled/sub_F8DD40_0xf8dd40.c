// Function: sub_F8DD40
// Address: 0xf8dd40
//
__int64 __fastcall sub_F8DD40(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 i; // r14
  __int64 v6; // r13
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 *v15; // rdi
  __int64 *v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 v22; // r13
  _BYTE *v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  _BYTE *v26; // r10
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // r15
  _BYTE *v29; // r13
  _BYTE *v30; // rax
  unsigned __int8 v31; // al
  unsigned int v32; // r13d
  unsigned __int64 *v33; // rdi
  __int64 v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // r13
  __int64 v39; // rax
  _BYTE *v40; // r14
  unsigned __int64 j; // r15
  _BYTE *v42; // rax
  int v44; // eax
  __int16 v45; // r12
  unsigned __int64 v46; // rax
  int v47; // edx
  __int64 v48; // rsi
  char v49; // r8
  char v50; // r12
  _BYTE *v51; // rax
  int v52; // eax
  __int64 v53; // [rsp+10h] [rbp-E0h]
  unsigned __int64 *v54; // [rsp+10h] [rbp-E0h]
  __int64 v56; // [rsp+20h] [rbp-D0h]
  __int64 *v57; // [rsp+28h] [rbp-C8h]
  __int64 *v58; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+38h] [rbp-B8h]
  _BYTE v60[176]; // [rsp+40h] [rbp-B0h] BYREF

  v3 = sub_D95540(**(_QWORD **)(a2 + 32));
  v4 = *(_QWORD *)(a2 + 32);
  v56 = v3;
  v58 = (__int64 *)v60;
  v59 = 0x800000000LL;
  for ( i = v4 + 8LL * *(_QWORD *)(a2 + 40); v4 != i; LODWORD(v59) = v59 + 1 )
  {
    v6 = *(_QWORD *)(i - 8);
    v9 = sub_F83200((__int64)a1, v6);
    v10 = (unsigned int)v59;
    v11 = (unsigned int)v59 + 1LL;
    if ( v11 > HIDWORD(v59) )
    {
      sub_C8D5F0((__int64)&v58, v60, v11, 0x10u, v7, v8);
      v10 = (unsigned int)v59;
    }
    i -= 8;
    v12 = &v58[2 * v10];
    *v12 = v9;
    v12[1] = v6;
  }
  v13 = 0;
  sub_F86830((__int64)&v58, *(_QWORD *)(*a1 + 40));
  v14 = (__int64)v58;
  v57 = v58;
  v15 = v58;
  v16 = &v58[2 * (unsigned int)v59];
  if ( v58 != v16 )
  {
    while ( 1 )
    {
      v18 = v57[1];
      if ( !v13 )
      {
        v37 = *v57;
        v38 = 0;
        while ( v57[1] == v18 )
        {
          if ( v38 == 0x7FFFFFFFFFFFFFFFLL )
          {
            v13 = sub_F894B0((__int64)a1, v18);
            v40 = (_BYTE *)v13;
LABEL_42:
            for ( j = 2; j <= v38; j *= 2LL )
            {
              v14 = 17;
              v42 = (_BYTE *)sub_F810E0(a1, 0x11u, v40, v40, 0, 1);
              v40 = v42;
              if ( (j & v38) != 0 )
              {
                if ( v13 )
                {
                  v14 = 17;
                  v13 = sub_F810E0(a1, 0x11u, (_BYTE *)v13, v42, 0, 1);
                }
                else
                {
                  v13 = (__int64)v42;
                }
              }
            }
            goto LABEL_8;
          }
          v57 += 2;
          ++v38;
          if ( v57 == v16 || v37 != *v57 )
            break;
        }
        v14 = v18;
        v39 = sub_F894B0((__int64)a1, v18);
        if ( (v38 & 1) != 0 )
          v13 = v39;
        v40 = (_BYTE *)v39;
        if ( v38 > 1 )
          goto LABEL_42;
        goto LABEL_8;
      }
      if ( !sub_D96960(v18) )
        break;
      v17 = (_BYTE *)sub_AD6530(v56, v14);
      v14 = 15;
      v57 += 2;
      v13 = sub_F810E0(a1, 0xFu, v17, (_BYTE *)v13, 0, 1);
LABEL_8:
      v15 = v58;
      v16 = &v58[2 * (unsigned int)v59];
      if ( v16 == v57 )
        goto LABEL_51;
    }
    v19 = &v58[2 * (unsigned int)v59];
    v20 = v57[1];
    if ( v19 == v57 )
    {
      v23 = 0;
      sub_F894B0((__int64)a1, v20);
    }
    else
    {
      v21 = *v57;
      v22 = 0;
      while ( v57[1] == v20 )
      {
        if ( v22 == 0x7FFFFFFFFFFFFFFFLL )
        {
          v23 = (_BYTE *)sub_F894B0((__int64)a1, v20);
          v26 = v23;
          goto LABEL_17;
        }
        v57 += 2;
        ++v22;
        if ( v19 == v57 || v21 != *v57 )
          break;
      }
      v23 = 0;
      v24 = sub_F894B0((__int64)a1, v20);
      if ( (v22 & 1) != 0 )
        v23 = (_BYTE *)v24;
      v26 = (_BYTE *)v24;
      if ( v22 > 1 )
      {
LABEL_17:
        v53 = v13;
        v27 = 2;
        v28 = v22;
        v29 = v26;
        while ( 1 )
        {
          v30 = (_BYTE *)sub_F810E0(a1, 0x11u, v29, v29, 0, 1);
          v29 = v30;
          if ( (v27 & v28) == 0 )
            goto LABEL_19;
          if ( v23 )
          {
            v23 = (_BYTE *)sub_F810E0(a1, 0x11u, v23, v30, 0, 1);
LABEL_19:
            v27 *= 2LL;
            if ( v27 > v28 )
              goto LABEL_23;
          }
          else
          {
            v27 *= 2LL;
            v23 = v30;
            if ( v27 > v28 )
            {
LABEL_23:
              v13 = v53;
              break;
            }
          }
        }
      }
    }
    v31 = *(_BYTE *)v13;
    if ( *(_BYTE *)v13 > 0x15u )
    {
      v25 = (__int64)v23;
      v31 = *v23;
      v23 = (_BYTE *)v13;
      v13 = v25;
    }
    if ( v31 == 17 )
    {
      v32 = *(_DWORD *)(v13 + 32);
      v33 = (unsigned __int64 *)(v13 + 24);
      if ( v32 > 0x40 )
      {
        v44 = sub_C44630((__int64)v33);
        v33 = (unsigned __int64 *)(v13 + 24);
        if ( v44 == 1 )
          goto LABEL_55;
      }
      else
      {
        v34 = *(_QWORD *)(v13 + 24);
        if ( v34 )
        {
          v25 = v34 - 1;
          if ( (v34 & (v34 - 1)) == 0 )
          {
LABEL_55:
            v45 = *(_WORD *)(a2 + 28) & 7;
            if ( v32 > 0x40 )
            {
              v47 = v32 - sub_C444A0((__int64)v33);
              v48 = (unsigned int)(v47 - 1);
            }
            else if ( *v33 )
            {
              _BitScanReverse64(&v46, *v33);
              LODWORD(v46) = v46 ^ 0x3F;
              v47 = 64 - v46;
              v48 = (unsigned int)(63 - v46);
            }
            else
            {
              v48 = 0xFFFFFFFFLL;
              v47 = 0;
            }
            v49 = v45;
            v50 = v45 & 3;
            if ( v32 != v47 )
              v50 = v49;
            v51 = (_BYTE *)sub_AD64C0(v56, v48, 0);
            v14 = 25;
            v13 = sub_F810E0(a1, 0x19u, v23, v51, v50, 1);
            goto LABEL_8;
          }
        }
      }
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17 > 1 )
        goto LABEL_36;
    }
    else
    {
      v25 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
      if ( (unsigned int)v25 > 1 || v31 > 0x15u )
      {
LABEL_36:
        v14 = 17;
        v13 = sub_F810E0(a1, 0x11u, v23, (_BYTE *)v13, *(_BYTE *)(a2 + 28) & 7, 1);
        goto LABEL_8;
      }
    }
    v35 = sub_AD7630(v13, 1, v25);
    if ( v35 && *v35 == 17 )
    {
      v32 = *((_DWORD *)v35 + 8);
      v33 = (unsigned __int64 *)(v35 + 24);
      if ( v32 > 0x40 )
      {
        v54 = (unsigned __int64 *)(v35 + 24);
        v52 = sub_C44630((__int64)v33);
        v33 = v54;
        if ( v52 == 1 )
          goto LABEL_55;
      }
      else
      {
        v36 = *((_QWORD *)v35 + 3);
        if ( v36 && (v36 & (v36 - 1)) == 0 )
          goto LABEL_55;
      }
    }
    goto LABEL_36;
  }
LABEL_51:
  if ( v15 != (__int64 *)v60 )
    _libc_free(v15, v14);
  return v13;
}
