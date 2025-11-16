// Function: sub_30B3A80
// Address: 0x30b3a80
//
__int64 __fastcall sub_30B3A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  _BYTE *v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 *v10; // rsi
  unsigned __int64 *v11; // rcx
  int v12; // eax
  __int64 *v13; // rdx
  unsigned __int64 v14; // rdi
  void (__fastcall *v15)(unsigned __int64); // rax
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rdx
  __int64 v19; // rsi
  unsigned __int64 v20; // r8
  unsigned __int64 *v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  void (__fastcall *v26)(unsigned __int64); // rax
  int v27; // eax
  unsigned __int64 *v28; // [rsp+18h] [rbp-138h]
  _BYTE *v29; // [rsp+20h] [rbp-130h]
  unsigned __int64 *v30; // [rsp+20h] [rbp-130h]
  __int64 v31; // [rsp+28h] [rbp-128h]
  int v32; // [rsp+28h] [rbp-128h]
  char v33; // [rsp+37h] [rbp-119h]
  __int64 v34; // [rsp+38h] [rbp-118h]
  __int64 v35; // [rsp+40h] [rbp-110h]
  __int64 *v36; // [rsp+48h] [rbp-108h]
  __int64 *v37; // [rsp+50h] [rbp-100h]
  _BYTE *v38; // [rsp+58h] [rbp-F8h]
  char v39; // [rsp+67h] [rbp-E9h] BYREF
  unsigned __int64 v40; // [rsp+68h] [rbp-E8h] BYREF
  __int64 (__fastcall *v41)(__int64, __int64); // [rsp+70h] [rbp-E0h] BYREF
  char *v42; // [rsp+78h] [rbp-D8h]
  __int64 *v43; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v44; // [rsp+88h] [rbp-C8h]
  _BYTE v45[64]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v46; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v47; // [rsp+D8h] [rbp-78h]
  _BYTE v48[112]; // [rsp+E0h] [rbp-70h] BYREF

  LODWORD(v6) = a1;
  v43 = (__int64 *)v45;
  v44 = 0x800000000LL;
  v47 = 0x800000000LL;
  v41 = sub_30B3090;
  v46 = v48;
  v42 = &v39;
  sub_30B0A30(a2, (__int64)&v41, (__int64)&v43);
  v42 = &v39;
  v41 = sub_30B3090;
  sub_30B0A30(a3, (__int64)&v41, (__int64)&v46);
  v7 = v46;
  v36 = &v43[(unsigned int)v44];
  if ( v43 != v36 )
  {
    v37 = v43;
    v35 = a4 + 16;
    v8 = a1 + 40;
    do
    {
      v6 = *v37;
      v38 = &v7[8 * (unsigned int)v47];
      if ( v38 == v7 )
        goto LABEL_15;
      do
      {
        while ( 1 )
        {
          sub_2297CA0((__int64 *)&v40, v8, v6, *(_BYTE **)v7);
          if ( !v40 )
            goto LABEL_6;
          v9 = *(unsigned int *)(a4 + 8);
          v10 = (__int64 *)&v40;
          v11 = *(unsigned __int64 **)a4;
          v12 = *(_DWORD *)(a4 + 8);
          if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
          {
            if ( v11 > &v40 || &v40 >= &v11[v9] )
            {
              v34 = -1;
              v33 = 0;
            }
            else
            {
              v33 = 1;
              v34 = &v40 - v11;
            }
            v17 = (unsigned __int64 *)sub_C8D7D0(a4, v35, v9 + 1, 8u, (unsigned __int64 *)&v41, v9 + 1);
            v18 = *(unsigned __int64 **)a4;
            v11 = v17;
            v19 = *(unsigned int *)(a4 + 8);
            v20 = *(_QWORD *)a4 + v19 * 8;
            if ( *(_QWORD *)a4 == v20 )
              goto LABEL_37;
            v21 = &v17[v19];
            do
            {
              if ( v17 )
              {
                *v17 = *v18;
                *v18 = 0;
              }
              ++v17;
              ++v18;
            }
            while ( v17 != v21 );
            v22 = *(_QWORD *)a4;
            v20 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
            if ( *(_QWORD *)a4 == v20 )
            {
LABEL_37:
              v27 = (int)v41;
              if ( v35 != v20 )
              {
                v30 = v11;
                v32 = (int)v41;
                _libc_free(v20);
                v11 = v30;
                v27 = v32;
              }
              v9 = *(unsigned int *)(a4 + 8);
              *(_DWORD *)(a4 + 12) = v27;
              *(_QWORD *)a4 = v11;
              v10 = (__int64 *)&v11[v34];
              v12 = v9;
              if ( !v33 )
                v10 = (__int64 *)&v40;
              goto LABEL_9;
            }
            v31 = v6;
            v23 = *(_QWORD *)a4;
            v29 = v7;
            v24 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
            v28 = v11;
            while ( 1 )
            {
LABEL_33:
              v25 = *(_QWORD *)(v24 - 8);
              v24 -= 8;
              if ( !v25 )
                goto LABEL_32;
              v26 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v25 + 8LL);
              if ( v26 == sub_228A6E0 )
                break;
              ((void (__fastcall *)(unsigned __int64, __int64, unsigned __int64 *))v26)(v25, v22, v18);
              if ( v23 == v24 )
              {
LABEL_36:
                v6 = v31;
                v7 = v29;
                v11 = v28;
                v20 = *(_QWORD *)a4;
                goto LABEL_37;
              }
            }
            v22 = 40;
            j_j___libc_free_0(v25);
LABEL_32:
            if ( v23 == v24 )
              goto LABEL_36;
            goto LABEL_33;
          }
LABEL_9:
          v13 = (__int64 *)&v11[v9];
          if ( v13 )
          {
            *v13 = *v10;
            *v10 = 0;
            v12 = *(_DWORD *)(a4 + 8);
          }
          v14 = v40;
          *(_DWORD *)(a4 + 8) = v12 + 1;
          if ( !v14 )
            goto LABEL_6;
          v15 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v14 + 8LL);
          if ( v15 != sub_228A6E0 )
            break;
          j_j___libc_free_0(v14);
LABEL_6:
          v7 += 8;
          if ( v38 == v7 )
            goto LABEL_14;
        }
        ((void (__fastcall *)(unsigned __int64, __int64 *))v15)(v14, v10);
        v7 += 8;
      }
      while ( v38 != v7 );
LABEL_14:
      v7 = v46;
LABEL_15:
      ++v37;
    }
    while ( v36 != v37 );
  }
  LOBYTE(v6) = *(_DWORD *)(a4 + 8) != 0;
  if ( v7 != v48 )
    _libc_free((unsigned __int64)v7);
  if ( v43 != (__int64 *)v45 )
    _libc_free((unsigned __int64)v43);
  return (unsigned int)v6;
}
