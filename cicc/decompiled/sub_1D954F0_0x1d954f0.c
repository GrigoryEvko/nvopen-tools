// Function: sub_1D954F0
// Address: 0x1d954f0
//
void __fastcall sub_1D954F0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rax
  int v4; // r9d
  unsigned int v5; // r14d
  char *v6; // rax
  unsigned int *v7; // r14
  unsigned int *i; // r15
  __int64 v9; // rbx
  char *v10; // r8
  unsigned int v11; // eax
  _DWORD *v12; // rdx
  _BYTE *v13; // r13
  __int64 v14; // r12
  unsigned int v15; // ebx
  __int64 v16; // r15
  __int64 v17; // r14
  unsigned int v18; // esi
  _DWORD *v19; // rax
  __int16 *v20; // rsi
  __int16 v21; // di
  __int16 *v22; // rsi
  unsigned __int16 v23; // r10
  __int16 *v24; // r11
  unsigned int v25; // esi
  _DWORD *v26; // rdi
  __int16 v27; // di
  unsigned int v28; // eax
  _DWORD *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // [rsp+10h] [rbp-110h]
  _BYTE *v32; // [rsp+28h] [rbp-F8h]
  __int64 v33; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-E8h]
  __int64 v35; // [rsp+40h] [rbp-E0h]
  __int64 v36; // [rsp+48h] [rbp-D8h]
  __int64 v37; // [rsp+50h] [rbp-D0h]
  _BYTE *v38; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+68h] [rbp-B8h]
  _BYTE v40[32]; // [rsp+70h] [rbp-B0h] BYREF
  char *v41; // [rsp+90h] [rbp-90h]
  unsigned int v42; // [rsp+98h] [rbp-88h]
  _BYTE *v43; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v44; // [rsp+A8h] [rbp-78h]
  _BYTE v45[112]; // [rsp+B0h] [rbp-70h] BYREF

  v2 = *(__int64 (**)())(**(_QWORD **)(((__int64 (*)(void))sub_1E15F70)() + 16) + 112LL);
  if ( v2 == sub_1D00B10 )
  {
    v41 = 0;
    v38 = v40;
    v42 = 0;
    v39 = 0x800000000LL;
    BUG();
  }
  v3 = v2();
  v41 = 0;
  v42 = 0;
  v5 = *(_DWORD *)(v3 + 16);
  v38 = v40;
  v31 = v3;
  v39 = 0x800000000LL;
  if ( v5 )
  {
    v6 = (char *)_libc_calloc(v5, 1u);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v6 = 0;
    }
    v41 = v6;
    v42 = v5;
  }
  v7 = *(unsigned int **)(a2 + 8);
  for ( i = &v7[*(unsigned int *)(a2 + 16)]; i != v7; ++v7 )
  {
    v9 = *v7;
    v10 = &v41[v9];
    v11 = (unsigned __int8)v41[v9];
    if ( v11 < (unsigned int)v39 )
    {
      while ( 1 )
      {
        v12 = &v38[4 * v11];
        if ( (_DWORD)v9 == *v12 )
          break;
        v11 += 256;
        if ( (unsigned int)v39 <= v11 )
          goto LABEL_47;
      }
      if ( v12 != (_DWORD *)&v38[4 * (unsigned int)v39] )
        continue;
    }
LABEL_47:
    *v10 = v39;
    v30 = (unsigned int)v39;
    if ( (unsigned int)v39 >= HIDWORD(v39) )
    {
      sub_16CD150((__int64)&v38, v40, 0, 4, (int)v10, v4);
      v30 = (unsigned int)v39;
    }
    *(_DWORD *)&v38[4 * v30] = v9;
    LODWORD(v39) = v39 + 1;
  }
  v43 = v45;
  v44 = 0x400000000LL;
  sub_1DC2290(a2, a1, &v43);
  v13 = v43;
  v32 = &v43[16 * (unsigned int)v44];
  if ( v43 == v32 )
    goto LABEL_23;
  do
  {
    v14 = *((_QWORD *)v13 + 1);
    v15 = *(_DWORD *)v13;
    v16 = *(_QWORD *)(v14 + 16);
    v17 = sub_1E15F70(v16);
    if ( *(_BYTE *)v14 == 12 )
    {
      v28 = (unsigned __int8)v41[v15];
      if ( v28 < (unsigned int)v39 )
      {
        while ( 1 )
        {
          v29 = &v38[4 * v28];
          if ( v15 == *v29 )
            break;
          v28 += 256;
          if ( (unsigned int)v39 <= v28 )
            goto LABEL_46;
        }
        if ( v29 != (_DWORD *)&v38[4 * (unsigned int)v39] )
        {
          v33 = 0x20000000;
          v35 = 0;
          v34 = v15;
          v36 = 0;
          v37 = 0;
          sub_1E1A9C0(v16, v17, &v33);
        }
      }
LABEL_46:
      v33 = 805306368;
      v35 = 0;
      v34 = v15;
      v36 = 0;
      v37 = 0;
      sub_1E1A9C0(v16, v17, &v33);
    }
    else
    {
      v18 = (unsigned __int8)v41[v15];
      if ( v18 < (unsigned int)v39 )
      {
        while ( 1 )
        {
          v19 = &v38[4 * v18];
          if ( v15 == *v19 )
            break;
          v18 += 256;
          if ( (unsigned int)v39 <= v18 )
            goto LABEL_28;
        }
        if ( v19 != (_DWORD *)&v38[4 * (unsigned int)v39] )
        {
LABEL_20:
          v33 = 0x20000000;
          v35 = 0;
          v34 = v15;
          v36 = 0;
          v37 = 0;
          sub_1E1A9C0(v16, v17, &v33);
          goto LABEL_21;
        }
      }
      else
      {
LABEL_28:
        v19 = &v38[4 * (unsigned int)v39];
      }
      v20 = (__int16 *)(*(_QWORD *)(v31 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v31 + 8) + 24LL * v15 + 4));
      v21 = *v20;
      v22 = v20 + 1;
      v23 = v21 + v15;
      if ( !v21 )
        v22 = 0;
LABEL_31:
      v24 = v22;
      while ( v24 )
      {
        v25 = (unsigned __int8)v41[v23];
        if ( (unsigned int)v39 > v25 )
        {
          while ( 1 )
          {
            v26 = &v38[4 * v25];
            if ( v23 == *v26 )
              break;
            v25 += 256;
            if ( (unsigned int)v39 <= v25 )
              goto LABEL_38;
          }
          if ( v19 != v26 )
            goto LABEL_20;
        }
LABEL_38:
        v27 = *v24;
        v22 = 0;
        ++v24;
        v23 += v27;
        if ( !v27 )
          goto LABEL_31;
      }
    }
LABEL_21:
    v13 += 16;
  }
  while ( v32 != v13 );
  v32 = v43;
LABEL_23:
  if ( v32 != v45 )
    _libc_free((unsigned __int64)v32);
  _libc_free((unsigned __int64)v41);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
}
