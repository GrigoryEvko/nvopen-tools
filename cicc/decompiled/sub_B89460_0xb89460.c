// Function: sub_B89460
// Address: 0xb89460
//
__int64 __fastcall sub_B89460(__int64 *a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rcx
  char *v8; // rsi
  char *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rax
  void (__fastcall *v14)(__int64, __int64, _QWORD); // rbx
  unsigned int v15; // eax
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    v3 = *(_QWORD **)(*(_QWORD *)(a2 + 8) - 8LL);
    if ( (*(int (__fastcall **)(_QWORD *))(*v3 + 40LL))(v3) <= 3 )
      break;
    sub_B823C0(a2);
  }
  if ( (*(unsigned int (__fastcall **)(_QWORD *))(*v3 + 40LL))(v3) == 3 )
    return sub_B88F40((__int64)v3, a1, 1);
  v5 = sub_22077B0(568);
  v6 = v5;
  if ( !v5 )
  {
    v9 = *(char **)(a2 + 8);
    v8 = *(char **)a2;
    v7 = 336;
    if ( *(char **)a2 == v9 )
    {
      v11 = v3[1];
      v12 = 0;
      goto LABEL_10;
    }
    goto LABEL_8;
  }
  *(_QWORD *)(v5 + 8) = 0;
  v7 = v5 + 336;
  *(_QWORD *)(v5 + 416) = v5 + 432;
  *(_QWORD *)(v5 + 16) = &unk_4F818FF;
  *(_QWORD *)(v5 + 56) = v5 + 104;
  *(_QWORD *)(v5 + 112) = v5 + 160;
  *(_QWORD *)(v5 + 192) = v5 + 208;
  *(_QWORD *)(v5 + 200) = 0x1000000000LL;
  *(_QWORD *)(v5 + 424) = 0x1000000000LL;
  *(_DWORD *)(v5 + 88) = 1065353216;
  *(_DWORD *)(v5 + 144) = 1065353216;
  *(_DWORD *)(v5 + 24) = 4;
  *(_QWORD *)(v5 + 32) = 0;
  *(_QWORD *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 64) = 1;
  *(_QWORD *)(v5 + 72) = 0;
  *(_QWORD *)(v5 + 80) = 0;
  *(_QWORD *)(v5 + 96) = 0;
  *(_QWORD *)(v5 + 104) = 0;
  *(_QWORD *)(v5 + 120) = 1;
  *(_QWORD *)(v5 + 128) = 0;
  *(_QWORD *)(v5 + 136) = 0;
  *(_QWORD *)(v5 + 152) = 0;
  *(_QWORD *)(v5 + 160) = 0;
  *(_BYTE *)(v5 + 168) = 0;
  *(_QWORD *)(v5 + 184) = 0;
  *(_DWORD *)(v5 + 560) = 0;
  *(_QWORD *)(v5 + 384) = 1;
  *(_QWORD *)(v5 + 392) = 0;
  *(_QWORD *)(v5 + 400) = 0;
  *(_DWORD *)(v5 + 408) = 0;
  v8 = *(char **)a2;
  *(_OWORD *)(v5 + 336) = 0;
  *(_OWORD *)(v5 + 352) = 0;
  *(_OWORD *)(v5 + 368) = 0;
  *(_QWORD *)(v5 + 176) = &unk_49DAB30;
  v9 = *(char **)(a2 + 8);
  for ( *(_QWORD *)v6 = &unk_49DAA78; v9 != v8; *(_QWORD *)(v7 - 8) = v10 + 208 )
  {
LABEL_8:
    v10 = *((_QWORD *)v9 - 1);
    v9 -= 8;
    v7 += 8;
  }
  v11 = v3[1];
  v12 = v6 + 176;
LABEL_10:
  v13 = *(unsigned int *)(v11 + 120);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 124) )
  {
    v17 = v12;
    sub_C8D5F0(v11 + 112, v11 + 128, v13 + 1, 8);
    v13 = *(unsigned int *)(v11 + 120);
    v12 = v17;
  }
  v16 = v12;
  *(_QWORD *)(*(_QWORD *)(v11 + 112) + 8 * v13) = v12;
  ++*(_DWORD *)(v11 + 120);
  v14 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v6 + 64LL);
  v15 = (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 40LL))(v3);
  v14(v6, a2, v15);
  sub_B841D0((char **)a2, v16);
  v3 = (_QWORD *)v16;
  return sub_B88F40((__int64)v3, a1, 1);
}
