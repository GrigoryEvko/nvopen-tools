// Function: sub_278F8C0
// Address: 0x278f8c0
//
__int64 __fastcall sub_278F8C0(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // r14d
  unsigned int v5; // r12d
  __int64 v7; // rax
  _QWORD *v8; // rdi
  int v10; // r14d
  __int64 v11; // rax
  int v12; // eax
  int v13; // r11d
  __int64 v14; // r10
  _QWORD *v15; // r8
  unsigned int i; // ecx
  __int64 v17; // r13
  unsigned int v18; // r9d
  char v19; // al
  __int64 v20; // rdx
  size_t v21; // rdx
  int v22; // eax
  unsigned int *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  _QWORD *v26; // [rsp+8h] [rbp-F8h]
  int v27; // [rsp+14h] [rbp-ECh]
  _QWORD *v28; // [rsp+18h] [rbp-E8h]
  unsigned int v29; // [rsp+18h] [rbp-E8h]
  int v30; // [rsp+18h] [rbp-E8h]
  int v31; // [rsp+20h] [rbp-E0h]
  __int64 v32; // [rsp+20h] [rbp-E0h]
  unsigned int v33; // [rsp+20h] [rbp-E0h]
  unsigned int v34; // [rsp+28h] [rbp-D8h]
  unsigned int v35; // [rsp+28h] [rbp-D8h]
  __int64 v36; // [rsp+28h] [rbp-D8h]
  __int64 v37; // [rsp+30h] [rbp-D0h]
  unsigned int **v38; // [rsp+30h] [rbp-D0h]
  _QWORD *v39; // [rsp+30h] [rbp-D0h]
  __int64 v40; // [rsp+38h] [rbp-C8h]
  __int64 v41[2]; // [rsp+40h] [rbp-C0h] BYREF
  int v42; // [rsp+50h] [rbp-B0h]
  char v43; // [rsp+54h] [rbp-ACh]
  __int64 v44; // [rsp+58h] [rbp-A8h]
  char *v45; // [rsp+60h] [rbp-A0h]
  __int64 v46; // [rsp+68h] [rbp-98h]
  char v47; // [rsp+70h] [rbp-90h] BYREF
  __int64 v48; // [rsp+80h] [rbp-80h]
  int v49; // [rsp+90h] [rbp-70h] BYREF
  char v50; // [rsp+94h] [rbp-6Ch]
  __int64 v51; // [rsp+98h] [rbp-68h]
  _BYTE *v52; // [rsp+A0h] [rbp-60h]
  __int64 v53; // [rsp+A8h] [rbp-58h]
  _BYTE v54[16]; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v8 = *(_QWORD **)(a2 + 16);
    v42 = -1;
    v10 = v4 - 1;
    v40 = v7;
    v45 = &v47;
    v46 = 0x400000000LL;
    v53 = 0x400000000LL;
    v11 = *(unsigned int *)(a2 + 24);
    v43 = 0;
    v48 = 0;
    v44 = 0;
    v49 = -2;
    v50 = 0;
    v51 = 0;
    v52 = v54;
    v55 = 0;
    v41[0] = sub_939680(v8, (__int64)v8 + 4 * v11);
    v12 = sub_278D0E0((int *)a2, (__int64 *)(a2 + 8), v41);
    v13 = 1;
    v14 = 0;
    v15 = (_QWORD *)(a2 + 48);
    for ( i = v10 & v12; ; i = v10 & (v31 + v34) )
    {
      v17 = v40 + ((unsigned __int64)i << 6);
      v18 = *(_DWORD *)v17;
      if ( *(_DWORD *)a2 == *(_DWORD *)v17 )
      {
        if ( v18 > 0xFFFFFFFD )
          goto LABEL_19;
        v38 = *(unsigned int ***)(a2 + 8);
        if ( v38 == *(unsigned int ***)(v17 + 8) )
        {
          v20 = *(unsigned int *)(a2 + 24);
          if ( v20 == *(_DWORD *)(v17 + 24) )
          {
            v21 = 4 * v20;
            v35 = *(_DWORD *)v17;
            if ( !v21 )
              goto LABEL_28;
            v26 = v15;
            v27 = v13;
            v29 = i;
            v32 = v14;
            v22 = memcmp(*(const void **)(a2 + 16), *(const void **)(v17 + 16), v21);
            v14 = v32;
            i = v29;
            v13 = v27;
            v15 = v26;
            v18 = v35;
            if ( !v22 )
            {
LABEL_28:
              if ( !*(_QWORD *)(a2 + 48) && !*(_QWORD *)(v17 + 48) )
                goto LABEL_19;
              v30 = v13;
              v33 = i;
              v23 = *v38;
              v36 = v14;
              v39 = v15;
              v24 = sub_A7AD50(v15, v23, *(_QWORD *)(v17 + 48));
              v15 = v39;
              v14 = v36;
              v41[1] = v25;
              v41[0] = v24;
              i = v33;
              v13 = v30;
              if ( (_BYTE)v25 )
              {
LABEL_19:
                *a3 = v17;
                v5 = 1;
                goto LABEL_20;
              }
              v18 = *(_DWORD *)v17;
            }
          }
        }
      }
      if ( v18 == -1 )
        break;
      v28 = v15;
      v31 = v13;
      v34 = i;
      v37 = v14;
      v19 = sub_278A2A0(v17, (__int64)&v49);
      v15 = v28;
      if ( v37 || !v19 )
        v17 = v37;
      v14 = v17;
      v13 = v31 + 1;
    }
    if ( !v14 )
      v14 = v17;
    *a3 = v14;
    v5 = 0;
LABEL_20:
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return v5;
}
