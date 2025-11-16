// Function: sub_1390AA0
// Address: 0x1390aa0
//
void __fastcall sub_1390AA0(__int64 *a1)
{
  __int64 v1; // rcx
  __int64 v2; // r13
  __int64 v3; // r13
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned int v7; // r8d
  unsigned int i; // eax
  __int64 v9; // r9
  unsigned __int64 v10; // rdi
  unsigned int *v11; // rsi
  unsigned int *v12; // rax
  __int64 v13; // rax
  int *v14; // rdx
  _BOOL4 v15; // r8d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  _BOOL4 v21; // r8d
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned int *v24; // r13
  __int64 v25; // rax
  int *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // [rsp+18h] [rbp-D8h]
  unsigned int v29; // [rsp+24h] [rbp-CCh]
  _BOOL4 v30; // [rsp+24h] [rbp-CCh]
  _BOOL4 v31; // [rsp+24h] [rbp-CCh]
  __int64 v32; // [rsp+28h] [rbp-C8h]
  int *v33; // [rsp+28h] [rbp-C8h]
  int *v34; // [rsp+28h] [rbp-C8h]
  unsigned int v35; // [rsp+3Ch] [rbp-B4h] BYREF
  unsigned int *v36; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+48h] [rbp-A8h]
  _BYTE v38[64]; // [rsp+50h] [rbp-A0h] BYREF
  char v39[8]; // [rsp+90h] [rbp-60h] BYREF
  int v40; // [rsp+98h] [rbp-58h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-50h]
  int *v42; // [rsp+A8h] [rbp-48h]
  int *v43; // [rsp+B0h] [rbp-40h]
  __int64 v44; // [rsp+B8h] [rbp-38h]

  v1 = *a1;
  v2 = a1[1];
  v36 = (unsigned int *)v38;
  v37 = 0x1000000000LL;
  v3 = (v2 - v1) >> 4;
  v40 = 0;
  v41 = 0;
  v42 = &v40;
  v43 = &v40;
  v44 = 0;
  if ( (_DWORD)v3 )
  {
    v5 = 0;
    v6 = 0;
    v28 = (unsigned int)v3;
    while ( 1 )
    {
      v7 = v5;
      for ( i = *(_DWORD *)(v1 + 16 * v5); i != -1; i = *(_DWORD *)(v1 + 16LL * i) )
        v7 = i;
      v35 = v7;
      if ( v6 )
        break;
      v9 = (unsigned int)v37;
      v10 = (unsigned __int64)v36;
      v11 = &v36[(unsigned int)v37];
      if ( v36 != v11 )
      {
        v12 = v36;
        while ( *v12 != v7 )
        {
          if ( v11 == ++v12 )
            goto LABEL_25;
        }
        if ( v12 != v11 )
        {
LABEL_11:
          if ( ++v5 == v28 )
            goto LABEL_21;
          goto LABEL_12;
        }
      }
LABEL_25:
      if ( (unsigned int)v37 > 0xFuLL )
      {
        while ( 1 )
        {
          v24 = (unsigned int *)(v10 + 4 * v9 - 4);
          v25 = sub_B996D0((__int64)v39, v24);
          if ( v26 )
          {
            v21 = v25 || v26 == &v40 || *v24 < v26[8];
            v31 = v21;
            v34 = v26;
            v22 = sub_22077B0(40);
            *(_DWORD *)(v22 + 32) = *v24;
            sub_220F040(v31, v22, v34, &v40);
            ++v44;
          }
          v23 = v37 - 1;
          LODWORD(v37) = v23;
          if ( !v23 )
            break;
          v10 = (unsigned __int64)v36;
          v9 = v23;
        }
        v27 = sub_B996D0((__int64)v39, &v35);
        if ( v14 )
        {
          if ( !v27 && v14 != &v40 )
          {
            v15 = v35 < v14[8];
            goto LABEL_17;
          }
LABEL_16:
          v15 = 1;
LABEL_17:
          v30 = v15;
          v33 = v14;
          v16 = sub_22077B0(40);
          *(_DWORD *)(v16 + 32) = v35;
          sub_220F040(v30, v16, v33, &v40);
          ++v44;
        }
      }
      else
      {
        if ( (unsigned int)v37 >= HIDWORD(v37) )
        {
          sub_16CD150(&v36, v38, 0, 4);
          v7 = v35;
          v11 = &v36[(unsigned int)v37];
        }
        *v11 = v7;
        LODWORD(v37) = v37 + 1;
      }
      v1 = *a1;
      v17 = *a1 + 16LL * v35;
      v18 = *(unsigned int *)(v17 + 4);
      if ( (_DWORD)v18 == -1 )
        goto LABEL_11;
      do
      {
        v19 = 16 * v18;
        *(_QWORD *)(v1 + v19 + 8) |= *(_QWORD *)(v17 + 8);
        v1 = *a1;
        v17 = *a1 + v19;
        v18 = *(unsigned int *)(v17 + 4);
      }
      while ( (_DWORD)v18 != -1 );
      if ( ++v5 == v28 )
      {
LABEL_21:
        v20 = v41;
        goto LABEL_22;
      }
LABEL_12:
      v6 = v44;
    }
    v32 = v1;
    v29 = v7;
    v13 = sub_B996D0((__int64)v39, &v35);
    v1 = v32;
    if ( !v14 )
      goto LABEL_11;
    if ( !v13 && v14 != &v40 )
    {
      v15 = v14[8] > v29;
      goto LABEL_17;
    }
    goto LABEL_16;
  }
  v20 = 0;
LABEL_22:
  sub_138EFA0(v20);
  if ( v36 != (unsigned int *)v38 )
    _libc_free((unsigned __int64)v36);
}
