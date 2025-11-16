// Function: sub_18E6AF0
// Address: 0x18e6af0
//
void __fastcall sub_18E6AF0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r10
  __int64 i; // rcx
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 *v19; // rdi
  unsigned __int64 *v20; // rsi
  __int64 v21; // r8
  int v22; // eax
  __int64 v23; // r8
  __int64 v24; // rdx
  unsigned __int64 *v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // r15
  __int64 v28; // rbx
  unsigned int v29; // eax
  __int64 v30; // r13
  __int64 v31; // rdi
  unsigned int v32; // eax
  char *v33; // rdi
  __int64 v34; // r13
  unsigned __int64 v37; // [rsp+10h] [rbp-F0h]
  __int64 v38; // [rsp+18h] [rbp-E8h]
  __int64 v39; // [rsp+20h] [rbp-E0h]
  __int64 v40; // [rsp+28h] [rbp-D8h]
  __int64 v41; // [rsp+28h] [rbp-D8h]
  __int64 v42; // [rsp+28h] [rbp-D8h]
  char *v43[2]; // [rsp+30h] [rbp-D0h] BYREF
  _BYTE v44[128]; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 *v45; // [rsp+C0h] [rbp-40h]
  int v46; // [rsp+C8h] [rbp-38h]

  v6 = a3 - 1;
  v7 = a1;
  v8 = a1 + 160 * a2;
  v9 = (unsigned __int64)(a3 - 1) >> 63;
  v37 = a4;
  v10 = (__int64)(v9 + v6) >> 1;
  if ( a2 >= v10 )
  {
    v14 = a2;
    v23 = a1 + 160 * a2;
  }
  else
  {
    v40 = v10;
    v11 = a1 + 160 * a2;
    for ( i = a2; ; i = v14 )
    {
      v17 = 2 * (i + 1);
      v14 = v17 - 1;
      v18 = a1 + 320 * (i + 1);
      v8 = a1 + 160 * (v17 - 1);
      v19 = *(__int64 **)(v18 + 144);
      v20 = *(unsigned __int64 **)(v8 + 144);
      v21 = *v19;
      if ( *v19 == *v20 )
      {
        v38 = v11;
        v39 = i;
        v22 = sub_16A9900((__int64)(v19 + 3), v20 + 3);
        v11 = v38;
        i = v39;
        if ( v22 >= 0 )
        {
          v8 = v18;
          v14 = v17;
        }
      }
      else if ( *(_DWORD *)(v21 + 8) >> 8 >= *(_DWORD *)(*v20 + 8) >> 8 )
      {
        v8 = a1 + 320 * (i + 1);
        v14 = 2 * (i + 1);
      }
      v15 = 160 * i;
      sub_18E63F0(v11, (char **)v8, v9, i, v21, a6);
      v16 = a1 + v15;
      *(_QWORD *)(v16 + 144) = *(_QWORD *)(v8 + 144);
      *(_DWORD *)(v16 + 152) = *(_DWORD *)(v8 + 152);
      if ( v14 >= v40 )
        break;
      v11 = v8;
    }
    v23 = v8;
    v7 = a1;
  }
  if ( (a3 & 1) == 0 )
  {
    a4 = (unsigned __int64)(a3 - 2) >> 63;
    if ( (a3 - 2) / 2 == v14 )
    {
      v14 = 2 * v14 + 1;
      v34 = v7 + 160 * v14;
      sub_18E63F0(v8, (char **)v34, v9, a4, v23, a6);
      v23 = v34;
      *(_QWORD *)(v8 + 144) = *(_QWORD *)(v34 + 144);
      *(_DWORD *)(v8 + 152) = *(_DWORD *)(v34 + 152);
      v8 = v34;
    }
  }
  v43[0] = v44;
  v43[1] = (char *)0x800000000LL;
  v24 = *(unsigned int *)(v37 + 8);
  if ( (_DWORD)v24 )
  {
    v42 = v23;
    sub_18E63F0((__int64)v43, (char **)v37, v24, a4, v23, a6);
    v23 = v42;
  }
  v25 = *(unsigned __int64 **)(v37 + 144);
  v46 = *(_DWORD *)(v37 + 152);
  v45 = v25;
  v26 = (v14 - 1) / 2;
  v27 = v26;
  if ( v14 > a2 )
  {
    v28 = v14;
    while ( 1 )
    {
      v30 = v7 + 160 * v27;
      v31 = *(_QWORD *)(v30 + 144);
      if ( *(_QWORD *)v31 == *v25 )
      {
        v41 = v23;
        v32 = sub_16A9900(v31 + 24, v25 + 3);
        v23 = v41;
        v29 = v32 >> 31;
      }
      else
      {
        v24 = *(_DWORD *)(*(_QWORD *)v31 + 8LL) >> 8;
        LOBYTE(v29) = (unsigned int)v24 < *(_DWORD *)(*v25 + 8) >> 8;
      }
      v8 = v7 + 160 * v28;
      if ( !(_BYTE)v29 )
        break;
      sub_18E63F0(v23, (char **)(v7 + 160 * v27), v24, v26, v23, a6);
      v24 = v27 - 1;
      *(_QWORD *)(v8 + 144) = *(_QWORD *)(v30 + 144);
      *(_DWORD *)(v8 + 152) = *(_DWORD *)(v30 + 152);
      if ( a2 >= v27 )
      {
        v23 = v7 + 160 * v27;
        v8 = v23;
        break;
      }
      v25 = v45;
      v28 = v27;
      v23 = v7 + 160 * v27;
      v27 = (v27 - 1) / 2;
    }
  }
  sub_18E63F0(v23, v43, v24, v26, v23, a6);
  v33 = v43[0];
  *(_QWORD *)(v8 + 144) = v45;
  *(_DWORD *)(v8 + 152) = v46;
  if ( v33 != v44 )
    _libc_free((unsigned __int64)v33);
}
