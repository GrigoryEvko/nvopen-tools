// Function: sub_EA72E0
// Address: 0xea72e0
//
__int64 *__fastcall sub_EA72E0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5, int a6)
{
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdi
  int v20; // eax
  unsigned int v21; // r13d
  __int64 *v22; // r14
  __int64 v23; // rdx
  int v24; // eax
  unsigned int v25; // r13d
  __int64 *v26; // r14
  __int64 v27; // rdx
  int v28; // eax
  unsigned int v29; // r13d
  __int64 *v30; // r14
  __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // rdx
  __int64 *result; // rax
  unsigned int v35; // r13d
  __int64 *v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rax
  __int64 *v49; // rax

  sub_ECD6D0();
  *(_QWORD *)a1 = off_49E47A8;
  sub_1095B50(a1 + 40, a5);
  v10 = a6;
  *(_QWORD *)(a1 + 240) = a5;
  *(_QWORD *)(a1 + 224) = a3;
  *(_QWORD *)(a1 + 248) = a2;
  if ( !a6 )
    v10 = 1;
  *(_QWORD *)(a1 + 232) = a4;
  *(_WORD *)(a1 + 312) = 0;
  *(_DWORD *)(a1 + 304) = v10;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 296) = 0;
  *(_DWORD *)(a1 + 308) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0x1800000000LL;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 8;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  v11 = sub_22077B0(64);
  v12 = *(_QWORD *)(a1 + 400);
  *(_QWORD *)(a1 + 392) = v11;
  v13 = (__int64 *)(v11 + ((4 * v12 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v14 = sub_22077B0(440);
  *(_BYTE *)(a1 + 472) |= 1u;
  *v13 = v14;
  *(_QWORD *)(a1 + 416) = v14;
  *(_QWORD *)(a1 + 448) = v14;
  *(_QWORD *)(a1 + 408) = v14;
  *(_QWORD *)(a1 + 440) = v14;
  *(_QWORD *)(a1 + 528) = a1 + 544;
  *(_QWORD *)(a1 + 536) = 0x400000000LL;
  *(_QWORD *)(a1 + 768) = a1 + 784;
  *(_QWORD *)(a1 + 776) = 0x200000000LL;
  *(_QWORD *)(a1 + 840) = a1 + 824;
  *(_QWORD *)(a1 + 848) = a1 + 824;
  *(_QWORD *)(a1 + 864) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 424) = v14 + 440;
  *(_QWORD *)(a1 + 456) = v14 + 440;
  *(_QWORD *)(a1 + 888) = 0x1000000000LL;
  *(_QWORD *)(a1 + 432) = v13;
  *(_QWORD *)(a1 + 464) = v13;
  *(_QWORD *)(a1 + 476) = 0;
  *(_QWORD *)(a1 + 484) = 0;
  *(_QWORD *)(a1 + 492) = 0;
  *(_QWORD *)(a1 + 500) = 0;
  *(_QWORD *)(a1 + 508) = 0;
  *(_BYTE *)(a1 + 520) = 0;
  *(_DWORD *)(a1 + 824) = 0;
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = 0x1000000000LL;
  v15 = *(_QWORD *)(a1 + 248);
  *(_BYTE *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 256) = *(_QWORD *)(v15 + 48);
  *(_QWORD *)(a1 + 264) = *(_QWORD *)(v15 + 56);
  *(_QWORD *)(v15 + 48) = sub_EA3870;
  *(_QWORD *)(v15 + 56) = a1;
  v16 = *(_QWORD *)(**(_QWORD **)(a1 + 248) + 24LL * (unsigned int)(*(_DWORD *)(a1 + 304) - 1));
  sub_1095BD0(a1 + 40, *(_QWORD *)(v16 + 8), *(_QWORD *)(v16 + 16) - *(_QWORD *)(v16 + 8), 0, 1);
  *(_QWORD *)(a4 + 264) = a1 + 280;
  switch ( *a3 )
  {
    case 0:
      v38 = sub_EC8F70();
      v39 = *(_QWORD *)(a1 + 272);
      v19 = v38;
      *(_QWORD *)(a1 + 272) = v38;
      if ( v39 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 8LL))(v39);
        v19 = *(_QWORD *)(a1 + 272);
      }
      *(_BYTE *)(a1 + 868) = 1;
      break;
    case 1:
      v17 = sub_ECB2C0();
      goto LABEL_5;
    case 2:
      v17 = sub_ECD500();
      goto LABEL_5;
    case 3:
      v17 = sub_EC4BA0();
      goto LABEL_5;
    case 4:
      sub_C64ED0("Need to implement createSPIRVAsmParser for SPIRV format.", 1u);
    case 5:
      v17 = sub_ECFAB0();
      goto LABEL_5;
    case 6:
      v17 = sub_ECFCE0();
LABEL_5:
      v18 = *(_QWORD *)(a1 + 272);
      v19 = v17;
      *(_QWORD *)(a1 + 272) = v17;
      if ( v18 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
        v19 = *(_QWORD *)(a1 + 272);
      }
      break;
    case 7:
      sub_C64ED0("DXContainer is not supported yet", 1u);
    default:
      v19 = *(_QWORD *)(a1 + 272);
      break;
  }
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v19 + 16LL))(v19, a1);
  sub_EA4C30(a1);
  v20 = sub_C92610();
  v21 = sub_C92740(a1 + 896, "reg", 3u, v20);
  v22 = (__int64 *)(*(_QWORD *)(a1 + 896) + 8LL * v21);
  v23 = *v22;
  if ( *v22 )
  {
    if ( v23 != -8 )
      goto LABEL_9;
    --*(_DWORD *)(a1 + 912);
  }
  v47 = sub_C7D670(20, 8);
  *(_WORD *)(v47 + 16) = 25970;
  *(_BYTE *)(v47 + 18) = 103;
  *(_BYTE *)(v47 + 19) = 0;
  *(_QWORD *)v47 = 3;
  *(_DWORD *)(v47 + 8) = 0;
  *v22 = v47;
  ++*(_DWORD *)(a1 + 908);
  v48 = (__int64 *)(*(_QWORD *)(a1 + 896) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 896), v21));
  v23 = *v48;
  if ( !*v48 || v23 == -8 )
  {
    v49 = v48 + 1;
    do
    {
      do
        v23 = *v49++;
      while ( !v23 );
    }
    while ( v23 == -8 );
  }
LABEL_9:
  *(_DWORD *)(v23 + 8) = 1;
  v24 = sub_C92610();
  v25 = sub_C92740(a1 + 896, "frame_ptr_rel", 0xDu, v24);
  v26 = (__int64 *)(*(_QWORD *)(a1 + 896) + 8LL * v25);
  v27 = *v26;
  if ( *v26 )
  {
    if ( v27 != -8 )
      goto LABEL_11;
    --*(_DWORD *)(a1 + 912);
  }
  v44 = sub_C7D670(30, 8);
  strcpy((char *)(v44 + 16), "frame_ptr_rel");
  *(_QWORD *)v44 = 13;
  *(_DWORD *)(v44 + 8) = 0;
  *v26 = v44;
  ++*(_DWORD *)(a1 + 908);
  v45 = (__int64 *)(*(_QWORD *)(a1 + 896) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 896), v25));
  v27 = *v45;
  if ( !*v45 || v27 == -8 )
  {
    v46 = v45 + 1;
    do
    {
      do
        v27 = *v46++;
      while ( !v27 );
    }
    while ( v27 == -8 );
  }
LABEL_11:
  *(_DWORD *)(v27 + 8) = 2;
  v28 = sub_C92610();
  v29 = sub_C92740(a1 + 896, "subfield_reg", 0xCu, v28);
  v30 = (__int64 *)(*(_QWORD *)(a1 + 896) + 8LL * v29);
  v31 = *v30;
  if ( *v30 )
  {
    if ( v31 != -8 )
      goto LABEL_13;
    --*(_DWORD *)(a1 + 912);
  }
  v41 = sub_C7D670(29, 8);
  strcpy((char *)(v41 + 16), "subfield_reg");
  *(_QWORD *)v41 = 12;
  *(_DWORD *)(v41 + 8) = 0;
  *v30 = v41;
  ++*(_DWORD *)(a1 + 908);
  v42 = (__int64 *)(*(_QWORD *)(a1 + 896) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 896), v29));
  v31 = *v42;
  if ( *v42 == -8 || !v31 )
  {
    v43 = v42 + 1;
    do
    {
      do
        v31 = *v43++;
      while ( v31 == -8 );
    }
    while ( !v31 );
  }
LABEL_13:
  *(_DWORD *)(v31 + 8) = 3;
  v32 = sub_C92610();
  v33 = (unsigned int)sub_C92740(a1 + 896, "reg_rel", 7u, v32);
  result = *(__int64 **)(a1 + 896);
  v35 = v33;
  v36 = &result[v33];
  v37 = *v36;
  if ( *v36 )
  {
    if ( v37 != -8 )
      goto LABEL_15;
    --*(_DWORD *)(a1 + 912);
  }
  v40 = sub_C7D670(24, 8);
  *(_DWORD *)(v40 + 16) = 1600611698;
  *(_WORD *)(v40 + 20) = 25970;
  *(_BYTE *)(v40 + 22) = 108;
  *(_BYTE *)(v40 + 23) = 0;
  *(_QWORD *)v40 = 7;
  *(_DWORD *)(v40 + 8) = 0;
  *v36 = v40;
  ++*(_DWORD *)(a1 + 908);
  result = (__int64 *)(*(_QWORD *)(a1 + 896) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 896), v35));
  v37 = *result;
  if ( *result == -8 || !v37 )
  {
    ++result;
    do
    {
      do
        v37 = *result++;
      while ( !v37 );
    }
    while ( v37 == -8 );
  }
LABEL_15:
  *(_DWORD *)(v37 + 8) = 4;
  return result;
}
