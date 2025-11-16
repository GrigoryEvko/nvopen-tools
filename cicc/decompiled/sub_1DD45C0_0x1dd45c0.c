// Function: sub_1DD45C0
// Address: 0x1dd45c0
//
void __fastcall sub_1DD45C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 (*v3)(); // rax
  __int64 v4; // r13
  char v5; // zf
  __int64 v6; // rdx
  int v7; // eax
  int v8; // r14d
  signed int v9; // ebx
  _DWORD *v10; // rax
  _DWORD *v11; // rdx
  unsigned int v12; // edx
  __int64 v13; // rdi
  __int64 v14; // rax
  int *v15; // rsi
  unsigned __int64 *v16; // rax
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rax
  unsigned int v19; // edx
  _DWORD *v20; // r8
  __int64 v21; // rcx
  __int64 v22; // rdx
  int v23; // ebx
  __int64 v24; // r12
  int v25; // r13d
  __int64 v26; // rax
  unsigned __int8 v27; // al
  char v28; // [rsp+50h] [rbp-1F0h]
  char v29; // [rsp+54h] [rbp-1ECh]
  unsigned int v30; // [rsp+60h] [rbp-1E0h] BYREF
  int v31; // [rsp+64h] [rbp-1DCh] BYREF
  __int64 v32; // [rsp+68h] [rbp-1D8h] BYREF
  __int64 v33; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v34; // [rsp+78h] [rbp-1C8h]
  __int64 v35; // [rsp+80h] [rbp-1C0h] BYREF
  unsigned __int64 v36[2]; // [rsp+A0h] [rbp-1A0h] BYREF
  _BYTE v37[32]; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v38; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v39; // [rsp+D8h] [rbp-168h]
  __int64 v40; // [rsp+E0h] [rbp-160h] BYREF
  unsigned __int64 v41[2]; // [rsp+100h] [rbp-140h] BYREF
  _BYTE v42[32]; // [rsp+110h] [rbp-130h] BYREF
  __int64 v43; // [rsp+130h] [rbp-110h] BYREF
  __int64 v44; // [rsp+138h] [rbp-108h]
  __int64 v45; // [rsp+140h] [rbp-100h] BYREF
  unsigned __int64 v46[2]; // [rsp+160h] [rbp-E0h] BYREF
  _BYTE v47[32]; // [rsp+170h] [rbp-D0h] BYREF
  _BYTE *v48; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+198h] [rbp-A8h]
  _BYTE v50[72]; // [rsp+1A0h] [rbp-A0h] BYREF
  int v51; // [rsp+1E8h] [rbp-58h] BYREF
  __int64 v52; // [rsp+1F0h] [rbp-50h]
  int *v53; // [rsp+1F8h] [rbp-48h]
  int *v54; // [rsp+200h] [rbp-40h]
  __int64 v55; // [rsp+208h] [rbp-38h]

  v2 = a1;
  v3 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 48LL);
  if ( v3 == sub_1D90020 )
    BUG();
  v4 = *(_QWORD *)(a2 + 56);
  v32 = 0;
  v30 = 0;
  v5 = *(_DWORD *)(v3() + 8) == 1;
  v48 = v50;
  v49 = 0x1000000000LL;
  v51 = 0;
  v52 = 0;
  v53 = &v51;
  v54 = &v51;
  v55 = 0;
  v29 = v5;
  if ( *(int *)(v4 + 68) >= 0 )
  {
    v16 = (unsigned __int64 *)&v35;
    v33 = 0;
    v34 = 1;
    do
    {
      *(_DWORD *)v16 = 0x7FFFFFFF;
      v16 = (unsigned __int64 *)((char *)v16 + 4);
    }
    while ( v16 != v36 );
    v38 = 0;
    v39 = 1;
    v36[0] = (unsigned __int64)v37;
    v36[1] = 0x800000000LL;
    v17 = (unsigned __int64 *)&v40;
    do
    {
      *(_DWORD *)v17 = 0x7FFFFFFF;
      v17 = (unsigned __int64 *)((char *)v17 + 4);
    }
    while ( v17 != v41 );
    v43 = 0;
    v44 = 1;
    v41[0] = (unsigned __int64)v42;
    v41[1] = 0x800000000LL;
    v18 = (unsigned __int64 *)&v45;
    do
    {
      *(_DWORD *)v18 = 0x7FFFFFFF;
      v18 = (unsigned __int64 *)((char *)v18 + 4);
    }
    while ( v18 != v46 );
    v19 = *(_DWORD *)(v4 + 68);
    v46[0] = (unsigned __int64)v47;
    v46[1] = 0x800000000LL;
    v28 = v5;
    sub_1DD2790(a1, v4, v19, &v32, v5, &v30);
    v21 = *(_QWORD *)(v4 + 8);
    v22 = *(unsigned int *)(v4 + 32);
    v23 = -858993459 * ((*(_QWORD *)(v4 + 16) - v21) >> 3) - v22;
    if ( !v23 )
    {
LABEL_45:
      sub_1DD3CB0(v2, (__int64)&v33, (__int64)&v48, v4, v28, &v32, &v30);
      sub_1DD3CB0(v2, (__int64)&v38, (__int64)&v48, v4, v28, &v32, &v30);
      sub_1DD3CB0(v2, (__int64)&v43, (__int64)&v48, v4, v28, &v32, &v30);
      if ( (_BYTE *)v46[0] != v47 )
        _libc_free(v46[0]);
      if ( (v44 & 1) == 0 )
        j___libc_free_0(v45);
      if ( (_BYTE *)v41[0] != v42 )
        _libc_free(v41[0]);
      if ( (v39 & 1) == 0 )
        j___libc_free_0(v40);
      if ( (_BYTE *)v36[0] != v37 )
        _libc_free(v36[0]);
      if ( (v34 & 1) == 0 )
        j___libc_free_0(v35);
      goto LABEL_3;
    }
    v24 = v4;
    v25 = 0;
    while ( 1 )
    {
      v26 = v21 + 40LL * (unsigned int)(v22 + v25);
      if ( *(_QWORD *)(v26 + 8) == -1 || v25 == *(_DWORD *)(v24 + 68) )
        goto LABEL_37;
      v27 = *(_BYTE *)(v26 + 36);
      if ( v27 == 2 )
        break;
      if ( v27 <= 2u )
      {
        if ( v27 )
        {
          v31 = v25;
          sub_1DD42F0((__int64)&v33, &v31, v22, v21, v20);
        }
LABEL_37:
        if ( v23 == ++v25 )
          goto LABEL_44;
        goto LABEL_38;
      }
      v31 = v25++;
      sub_1DD42F0((__int64)&v43, &v31, v22, v21, v20);
      if ( v23 == v25 )
      {
LABEL_44:
        v4 = v24;
        v2 = a1;
        goto LABEL_45;
      }
LABEL_38:
      v22 = *(unsigned int *)(v24 + 32);
      v21 = *(_QWORD *)(v24 + 8);
    }
    v31 = v25;
    sub_1DD42F0((__int64)&v38, &v31, v22, v21, v20);
    goto LABEL_37;
  }
LABEL_3:
  v6 = *(_QWORD *)(v4 + 8);
  v7 = *(_DWORD *)(v4 + 32);
  v8 = -858993459 * ((*(_QWORD *)(v4 + 16) - v6) >> 3) - v7;
  if ( v8 )
  {
    v9 = 0;
    while ( *(_QWORD *)(v6 + 40LL * (unsigned int)(v9 + v7) + 8) != -1 && v9 != *(_DWORD *)(v4 + 68) )
    {
      if ( v55 )
      {
        v14 = v52;
        if ( v52 )
        {
          v15 = &v51;
          do
          {
            if ( v9 > *(_DWORD *)(v14 + 32) )
            {
              v14 = *(_QWORD *)(v14 + 24);
            }
            else
            {
              v15 = (int *)v14;
              v14 = *(_QWORD *)(v14 + 16);
            }
          }
          while ( v14 );
          if ( v15 != &v51 && v9 >= v15[8] )
            break;
        }
      }
      else
      {
        v10 = v48;
        v11 = &v48[4 * (unsigned int)v49];
        if ( v48 != (_BYTE *)v11 )
        {
          while ( v9 != *v10 )
          {
            if ( v11 == ++v10 )
              goto LABEL_17;
          }
          if ( v10 != v11 )
            break;
        }
      }
LABEL_17:
      v12 = v9++;
      sub_1DD2790(v2, v4, v12, &v32, v29 & 1, &v30);
      if ( v9 == v8 )
        goto LABEL_18;
LABEL_14:
      v7 = *(_DWORD *)(v4 + 32);
      v6 = *(_QWORD *)(v4 + 8);
    }
    if ( ++v9 == v8 )
      goto LABEL_18;
    goto LABEL_14;
  }
LABEL_18:
  v13 = v52;
  *(_QWORD *)(v4 + 640) = v32;
  *(_DWORD *)(v4 + 648) = v30;
  sub_1DD2B10(v13);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
}
