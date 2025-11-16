// Function: sub_2DD4CA0
// Address: 0x2dd4ca0
//
void __fastcall sub_2DD4CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 v16; // rax
  char *v17; // rbx
  int v18; // r15d
  __int64 v19; // r14
  __int64 v20; // rdi
  __int64 v21; // r14
  __int64 *v22; // r15
  int v23; // ebx
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  char *v34; // rdi
  __int64 v35; // [rsp+0h] [rbp-A0h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+8h] [rbp-98h]
  __int64 v38; // [rsp+10h] [rbp-90h]
  __int64 v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+10h] [rbp-90h]
  unsigned int v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+18h] [rbp-88h]
  unsigned int v43; // [rsp+18h] [rbp-88h]
  char *v44[2]; // [rsp+20h] [rbp-80h] BYREF
  _BYTE v45[48]; // [rsp+30h] [rbp-70h] BYREF
  int v46; // [rsp+60h] [rbp-40h]
  int v47; // [rsp+68h] [rbp-38h]

  while ( 1 )
  {
    v42 = a3;
    if ( !a4 || !a5 )
      break;
    v5 = a1;
    v6 = a4;
    if ( a4 + a5 == 2 )
    {
      v17 = *(char **)a2;
      v18 = 0;
      v19 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v19 )
      {
        do
        {
          v20 = *(_QWORD *)v17;
          v17 += 8;
          v18 += sub_39FAC40(v20);
        }
        while ( v17 != (char *)v19 );
      }
      v41 = *(_DWORD *)(v5 + 8);
      v21 = *(_QWORD *)v5 + 8LL * v41;
      v43 = *(_DWORD *)(a2 + 72) * v18;
      if ( *(_QWORD *)v5 != v21 )
      {
        v22 = *(__int64 **)v5;
        v23 = 0;
        do
        {
          v24 = *v22++;
          v23 += sub_39FAC40(v24);
        }
        while ( (__int64 *)v21 != v22 );
        v28 = *(_DWORD *)(v5 + 72);
        if ( v43 < v28 * v23 )
        {
          v44[0] = v45;
          v44[1] = (char *)0x600000000LL;
          if ( v41 )
          {
            sub_2DD3500((__int64)v44, (char **)v5, v41, v25, v26, v27);
            v28 = *(_DWORD *)(v5 + 72);
          }
          v29 = *(unsigned int *)(v5 + 64);
          v47 = v28;
          v46 = v29;
          sub_2DD3500(v5, (char **)a2, v29, v25, v26, v27);
          *(_DWORD *)(v5 + 64) = *(_DWORD *)(a2 + 64);
          *(_DWORD *)(v5 + 72) = *(_DWORD *)(a2 + 72);
          sub_2DD3500(a2, v44, v30, v31, v32, v33);
          v34 = v44[0];
          *(_DWORD *)(a2 + 64) = v46;
          *(_DWORD *)(a2 + 72) = v47;
          if ( v34 != v45 )
            _libc_free((unsigned __int64)v34);
        }
      }
      return;
    }
    v36 = a5;
    if ( a4 > a5 )
    {
      v14 = a4 / 2;
      v40 = a1 + 80 * (a4 / 2);
      v16 = sub_2DD3790(a2, a3, v40);
      v13 = v40;
      v11 = v36;
      v12 = v16;
      v7 = 0xCCCCCCCCCCCCCCCDLL * ((v16 - a2) >> 4);
    }
    else
    {
      v7 = a5 / 2;
      v38 = a2 + 80 * (a5 / 2);
      v8 = sub_2DD3660(a1, a2, v38);
      v11 = v36;
      v12 = v38;
      v13 = v8;
      v14 = 0xCCCCCCCCCCCCCCCDLL * ((v8 - a1) >> 4);
    }
    v35 = v11;
    v37 = v12;
    v39 = v13;
    v15 = sub_2DD3C50(v13, a2, v12, v9, v10, v11);
    sub_2DD4CA0(a1, v39, v15, v14, v7);
    a1 = v15;
    a3 = v42;
    a4 = v6 - v14;
    a5 = v35 - v7;
    a2 = v37;
  }
}
