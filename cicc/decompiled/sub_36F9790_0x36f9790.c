// Function: sub_36F9790
// Address: 0x36f9790
//
__int64 __fastcall sub_36F9790(__int64 *a1, unsigned int a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (*v9)(void); // rax
  const char *v10; // rax
  size_t v11; // rdx
  _WORD *v12; // rdi
  unsigned __int8 *v13; // rsi
  unsigned __int64 v14; // rax
  __int64 *v15; // r15
  __int64 v16; // rbx
  __int64 *v17; // r13
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned int v21; // r13d
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r15
  _DWORD *v26; // rbx
  _DWORD *v27; // r12
  __int64 v28; // r8
  unsigned int v29; // edi
  __int64 v30; // rsi
  __int64 *v31; // rbx
  __int64 *v32; // r12
  __int64 v33; // rdi
  __int64 v36; // [rsp+20h] [rbp-1E0h]
  size_t v37; // [rsp+28h] [rbp-1D8h]
  __int64 v38; // [rsp+30h] [rbp-1D0h]
  __int64 v39; // [rsp+38h] [rbp-1C8h]
  __int64 v40; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 *v41; // [rsp+48h] [rbp-1B8h]
  __int64 v42; // [rsp+50h] [rbp-1B0h]
  __int64 v43; // [rsp+58h] [rbp-1A8h]
  _BYTE *v44; // [rsp+60h] [rbp-1A0h] BYREF
  size_t v45; // [rsp+68h] [rbp-198h]
  _BYTE v46[16]; // [rsp+70h] [rbp-190h] BYREF
  __int64 v47[2]; // [rsp+80h] [rbp-180h] BYREF
  void *v48; // [rsp+90h] [rbp-170h]
  __int64 v49; // [rsp+98h] [rbp-168h]
  void *dest; // [rsp+A0h] [rbp-160h]
  __int64 v51; // [rsp+A8h] [rbp-158h]
  _BYTE **v52; // [rsp+B0h] [rbp-150h]
  _BYTE *v53; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v54; // [rsp+C8h] [rbp-138h]
  _BYTE v55[304]; // [rsp+D0h] [rbp-130h] BYREF

  v53 = v55;
  v54 = 0x1000000000LL;
  v6 = a1[4];
  v40 = 0;
  v36 = v6;
  v7 = a1[5];
  v8 = a1[2];
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v38 = v7;
  v9 = *(__int64 (**)(void))(*(_QWORD *)v8 + 128LL);
  if ( (char *)v9 == (char *)sub_30594F0 )
    v39 = v8 + 376;
  else
    v39 = v9();
  v44 = v46;
  v51 = 0x100000000LL;
  v45 = 0;
  v46[0] = 0;
  v47[0] = (__int64)&unk_49DD210;
  v47[1] = 0;
  v48 = 0;
  v49 = 0;
  dest = 0;
  v52 = &v44;
  sub_CB5980((__int64)v47, 0, 0, 0);
  v10 = sub_2E791E0(a1);
  v12 = dest;
  v13 = (unsigned __int8 *)v10;
  v14 = v49 - (_QWORD)dest;
  if ( v11 > v49 - (__int64)dest )
  {
    v24 = sub_CB6200((__int64)v47, v13, v11);
    v12 = *(_WORD **)(v24 + 32);
    v15 = (__int64 *)v24;
    v14 = *(_QWORD *)(v24 + 24) - (_QWORD)v12;
  }
  else
  {
    v15 = v47;
    if ( v11 )
    {
      v37 = v11;
      memcpy(dest, v13, v11);
      v23 = v49 - ((_QWORD)dest + v37);
      dest = (char *)dest + v37;
      v12 = dest;
      if ( v23 > 6 )
        goto LABEL_6;
      goto LABEL_30;
    }
  }
  if ( v14 > 6 )
  {
LABEL_6:
    *(_DWORD *)v12 = 1918988383;
    v12[2] = 28001;
    *((_BYTE *)v12 + 6) = 95;
    v15[4] += 7;
    goto LABEL_7;
  }
LABEL_30:
  v15 = (__int64 *)sub_CB6200((__int64)v15, "_param_", 7u);
LABEL_7:
  sub_CB59D0((__int64)v15, a2);
  if ( dest != v48 )
    sub_CB5AE0(v47);
  v16 = a1[41];
  v17 = a1 + 40;
  if ( (__int64 *)v16 == v17 )
  {
LABEL_18:
    v21 = 0;
    sub_36F7DA0(v38, v44, v45);
  }
  else
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v16 + 56);
      v19 = v16 + 48;
      if ( v18 != v16 + 48 )
        break;
LABEL_17:
      v16 = *(_QWORD *)(v16 + 8);
      if ( v17 == (__int64 *)v16 )
        goto LABEL_18;
    }
    while ( 1 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(v18 + 68) - 2676 <= 1 )
      {
        v20 = *(_QWORD *)(v18 + 32);
        if ( *(_BYTE *)(v20 + 280) == 9 && !sub_2241AC0((__int64)&v44, *(const char **)(v20 + 304)) )
          break;
      }
      if ( (*(_BYTE *)v18 & 4) != 0 )
      {
        v18 = *(_QWORD *)(v18 + 8);
        if ( v18 == v19 )
          goto LABEL_17;
      }
      else
      {
        while ( (*(_BYTE *)(v18 + 44) & 8) != 0 )
          v18 = *(_QWORD *)(v18 + 8);
        v18 = *(_QWORD *)(v18 + 8);
        if ( v18 == v19 )
          goto LABEL_17;
      }
    }
    v21 = sub_36F8D10(v18, a3, (__int64)&v40, a4, (__int64 *)&v53, v36);
    if ( (_BYTE)v21 )
    {
      sub_36F7DA0(v38, v44, v45);
      v25 = *(_QWORD *)(*(_QWORD *)(v18 + 32) + 304LL);
      if ( (_DWORD)v42 )
      {
        v31 = v41;
        v32 = &v41[(unsigned int)v43];
        if ( v41 != v32 )
        {
          while ( *v31 == -8192 || *v31 == -4096 )
          {
            if ( v32 == ++v31 )
              goto LABEL_33;
          }
          while ( v32 != v31 )
          {
            v33 = *v31++;
            sub_2EAB400(v33, v25, 0);
            if ( v31 == v32 )
              break;
            while ( *v31 == -8192 || *v31 == -4096 )
            {
              if ( v32 == ++v31 )
                goto LABEL_33;
            }
          }
        }
      }
LABEL_33:
      v26 = v53;
      v27 = &v53[16 * (unsigned int)v54];
      if ( v27 != (_DWORD *)v53 )
      {
        do
        {
          v28 = *(_QWORD *)v26;
          v29 = *(unsigned __int16 *)(*(_QWORD *)v26 + 68LL);
          switch ( v26[2] )
          {
            case 0:
              v30 = -40LL * (unsigned int)sub_36F6540(v29);
              break;
            case 1:
              v30 = -40LL * (unsigned int)sub_36F6940(v29);
              break;
            case 2:
              v30 = -40LL * (unsigned int)sub_36F6E50(v29);
              break;
            case 3:
              v30 = -40LL * (unsigned int)sub_36F75C0(v29);
              break;
            case 4:
              v30 = -40LL * (unsigned int)sub_36F7480(v29);
              break;
            default:
              v30 = 0;
              break;
          }
          v26 += 4;
          sub_2E88D70(v28, (unsigned __int16 *)(*(_QWORD *)(v39 + 8) + v30));
        }
        while ( v27 != v26 );
      }
    }
  }
  v47[0] = (__int64)&unk_49DD210;
  sub_CB5840((__int64)v47);
  if ( v44 != v46 )
    j_j___libc_free_0((unsigned __int64)v44);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  sub_C7D6A0((__int64)v41, 8LL * (unsigned int)v43, 8);
  return v21;
}
