// Function: sub_27362B0
// Address: 0x27362b0
//
void __fastcall sub_27362B0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rdx
  unsigned __int64 *v12; // r14
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r13
  unsigned __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rax
  _QWORD *v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r9
  unsigned int v29; // [rsp+8h] [rbp-138h]
  unsigned int v30; // [rsp+10h] [rbp-130h]
  int v31; // [rsp+10h] [rbp-130h]
  int v32; // [rsp+18h] [rbp-128h]
  __int64 v34; // [rsp+38h] [rbp-108h] BYREF
  __int64 *v35; // [rsp+40h] [rbp-100h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-F8h]
  __int64 v37; // [rsp+50h] [rbp-F0h] BYREF
  int v38; // [rsp+58h] [rbp-E8h] BYREF
  _QWORD *v39; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v40; // [rsp+68h] [rbp-D8h]
  _QWORD v41[2]; // [rsp+70h] [rbp-D0h] BYREF
  char v42; // [rsp+80h] [rbp-C0h]
  __int64 v43; // [rsp+F0h] [rbp-50h]
  __int64 v44; // [rsp+F8h] [rbp-48h]
  int v45; // [rsp+100h] [rbp-40h]

  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a5 + 8) + 8LL) - 17 > 1 )
  {
    v6 = *(_QWORD *)(a5 - 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF));
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)v6 == 3 )
    {
      v34 = *(_QWORD *)(a5 - 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF));
      v8 = sub_AE4540(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24), *(_DWORD *)(*(_QWORD *)(v6 + 8) + 8LL) >> 8);
      v39 = (_QWORD *)sub_9208B0(*(_QWORD *)(a1 + 32), v8);
      v40 = v9;
      v36 = sub_CA1930(&v39);
      if ( v36 > 0x40 )
        sub_C43690((__int64)&v35, 0, 1);
      else
        v35 = 0;
      if ( (*(_BYTE *)(a5 + 1) & 2) == 0 || !(unsigned __int8)sub_BB6360(a5, *(_QWORD *)(a1 + 32), (__int64)&v35, 0, 0) )
        goto LABEL_8;
      if ( v36 > 0x40 )
      {
        v30 = v36;
        if ( v30 - (unsigned int)sub_C444A0((__int64)&v35) > 0x20 )
        {
LABEL_9:
          if ( v35 )
            j_j___libc_free_0_0((unsigned __int64)v35);
          return;
        }
        goto LABEL_14;
      }
      if ( !v35 || (_BitScanReverse64(&v22, (unsigned __int64)v35), 64 - ((unsigned int)v22 ^ 0x3F) <= 0x20) )
      {
LABEL_14:
        v10 = sub_DFB040(*(__int64 **)a1);
        v31 = v11;
        v32 = v10;
        v12 = (unsigned __int64 *)sub_2735BC0(a1 + 88, &v34, v11);
        v38 = 0;
        v37 = a5 | 4;
        sub_2733220((__int64)&v39, a2, &v37, &v38);
        v15 = v41[0];
        if ( v42 )
        {
          v29 = v36;
          if ( v36 > 0x40 )
          {
            v23 = -1;
            if ( v29 - (unsigned int)sub_C444A0((__int64)&v35) <= 0x40 )
              v23 = *v35;
          }
          else
          {
            v23 = (__int64)v35;
          }
          v24 = sub_BCB2D0(*(_QWORD **)(a1 + 24));
          v25 = sub_ACD640(v24, v23, 0);
          v44 = a5;
          v45 = 0;
          v39 = v41;
          v40 = 0x800000000LL;
          v43 = v25;
          sub_2731550(v12, (__int64)&v39, v26, v27, v25, v28);
          if ( v39 != v41 )
            _libc_free((unsigned __int64)v39);
          v16 = 1022611261 * (unsigned int)((__int64)(v12[1] - *v12) >> 3) - 1;
          *(_DWORD *)(v15 + 8) = v16;
        }
        else
        {
          v16 = *(unsigned int *)(v41[0] + 8LL);
        }
        v17 = *v12 + 168 * v16;
        v18 = *(unsigned int *)(v17 + 12);
        v19 = 0;
        if ( !v31 )
          v19 = v32;
        *(_DWORD *)(v17 + 160) += v19;
        v20 = *(unsigned int *)(v17 + 8);
        if ( v20 + 1 > v18 )
        {
          sub_C8D5F0(v17, (const void *)(v17 + 16), v20 + 1, 0x10u, v13, v14);
          v20 = *(unsigned int *)(v17 + 8);
        }
        v21 = (_QWORD *)(*(_QWORD *)v17 + 16 * v20);
        *v21 = a3;
        v21[1] = a4;
        ++*(_DWORD *)(v17 + 8);
LABEL_8:
        if ( v36 <= 0x40 )
          return;
        goto LABEL_9;
      }
    }
  }
}
