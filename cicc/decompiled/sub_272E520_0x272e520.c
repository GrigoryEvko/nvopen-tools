// Function: sub_272E520
// Address: 0x272e520
//
void __fastcall sub_272E520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // r11
  __int64 v18; // r10
  __int64 v19; // r15
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  char *v30; // rdi
  __int64 v31; // [rsp-108h] [rbp-108h]
  __int64 v32; // [rsp-100h] [rbp-100h]
  __int64 v33; // [rsp-100h] [rbp-100h]
  __int64 v34; // [rsp-F8h] [rbp-F8h]
  __int64 v35; // [rsp-F8h] [rbp-F8h]
  __int64 v36; // [rsp-F8h] [rbp-F8h]
  __int64 v37; // [rsp-F0h] [rbp-F0h]
  __int64 v38; // [rsp-F0h] [rbp-F0h]
  __int64 v39; // [rsp-F0h] [rbp-F0h]
  char *v40[2]; // [rsp-E8h] [rbp-E8h] BYREF
  _BYTE v41[128]; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v42; // [rsp-58h] [rbp-58h]
  __int64 v43; // [rsp-50h] [rbp-50h]
  int v44; // [rsp-48h] [rbp-48h]
  __int64 v45; // [rsp-30h] [rbp-30h]
  __int64 v46; // [rsp-20h] [rbp-20h]
  __int64 v47; // [rsp-18h] [rbp-18h]
  __int64 v48; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v48 = v9;
    v47 = v8;
    v46 = v7;
    v10 = a5;
    v45 = v6;
    if ( !a5 )
      break;
    v11 = a1;
    v12 = a4;
    if ( a4 + a5 == 2 )
    {
      v22 = *(_QWORD *)(a1 + 144);
      v23 = *(_QWORD *)(a2 + 144);
      if ( *(_QWORD *)(v23 + 8) == *(_QWORD *)(v22 + 8) )
      {
        if ( (int)sub_C49970(v23 + 24, (unsigned __int64 *)(v22 + 24)) >= 0 )
          return;
      }
      else if ( *(_DWORD *)(v23 + 32) >= *(_DWORD *)(v22 + 32) )
      {
        return;
      }
      v40[1] = (char *)0x800000000LL;
      v24 = *(_DWORD *)(v11 + 8);
      v40[0] = v41;
      if ( v24 )
      {
        sub_272D8A0((__int64)v40, (char **)v11, a3, a4, a5, a6);
        v22 = *(_QWORD *)(v11 + 144);
      }
      v25 = *(_QWORD *)(v11 + 152);
      v42 = v22;
      v43 = v25;
      v44 = *(_DWORD *)(v11 + 160);
      sub_272D8A0(v11, (char **)a2, a3, a4, a5, a6);
      *(_QWORD *)(v11 + 144) = *(_QWORD *)(a2 + 144);
      *(_QWORD *)(v11 + 152) = *(_QWORD *)(a2 + 152);
      *(_DWORD *)(v11 + 160) = *(_DWORD *)(a2 + 160);
      sub_272D8A0(a2, v40, v26, v27, v28, v29);
      v30 = v40[0];
      *(_QWORD *)(a2 + 144) = v42;
      *(_QWORD *)(a2 + 152) = v43;
      *(_DWORD *)(a2 + 160) = v44;
      if ( v30 != v41 )
        _libc_free((unsigned __int64)v30);
      return;
    }
    if ( a4 > a5 )
    {
      v39 = a3;
      v19 = a4 / 2;
      v36 = a1 + 168 * (a4 / 2);
      v21 = sub_272DA00(a2, a3, v36);
      v15 = v39;
      v17 = v36;
      v18 = v21;
      v16 = 0xCF3CF3CF3CF3CF3DLL * ((v21 - a2) >> 3);
    }
    else
    {
      v32 = a3;
      v34 = a5 / 2;
      v37 = a2 + 168 * (a5 / 2);
      v13 = sub_272DAD0(a1, a2, v37);
      v15 = v32;
      v16 = v34;
      v17 = v13;
      v18 = v37;
      v19 = 0xCF3CF3CF3CF3CF3DLL * ((v13 - a1) >> 3);
    }
    v31 = v15;
    v33 = v18;
    v38 = v16;
    v35 = v17;
    v20 = sub_272E010(v17, a2, v18, v14, v16, v15);
    sub_272E520(a1, v35, v20, v19, v38);
    a4 = v12 - v19;
    a1 = v20;
    a6 = v31;
    v6 = v45;
    a5 = v10 - v38;
    a3 = v31;
    a2 = v33;
    v7 = v46;
    v8 = v47;
    v9 = v48;
  }
}
