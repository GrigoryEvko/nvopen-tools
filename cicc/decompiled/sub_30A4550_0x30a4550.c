// Function: sub_30A4550
// Address: 0x30a4550
//
__int64 __fastcall sub_30A4550(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // eax
  __int64 v4; // rbx
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 *v8; // r13
  __int64 v9; // r8
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  if ( v2 != *(_QWORD *)a2 )
  {
    while ( (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(v2 - 8) + 40LL))(*(_QWORD *)(v2 - 8)) > 2 )
    {
      sub_B823C0(a2);
      v2 = *(_QWORD *)(a2 + 8);
      if ( *(_QWORD *)a2 == v2 )
        goto LABEL_6;
    }
    v2 = *(_QWORD *)(a2 + 8);
  }
LABEL_6:
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 - 8) + 40LL))(*(_QWORD *)(v2 - 8));
  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 8) - 8LL);
  if ( v3 == 2 )
  {
    if ( !v4 )
      v4 = 176;
  }
  else
  {
    v6 = sub_22077B0(0x238u);
    v8 = (__int64 *)v6;
    if ( v6 )
    {
      *(_QWORD *)(v6 + 8) = 0;
      *(_QWORD *)(v6 + 16) = &unk_502E0CE;
      v9 = v6 + 176;
      *(_QWORD *)(v6 + 56) = v6 + 104;
      *(_QWORD *)(v6 + 112) = v6 + 160;
      *(_QWORD *)(v6 + 192) = v6 + 208;
      *(_QWORD *)(v6 + 200) = 0x1000000000LL;
      *(_QWORD *)(v6 + 424) = 0x1000000000LL;
      *(_DWORD *)(v6 + 88) = 1065353216;
      *(_DWORD *)(v6 + 144) = 1065353216;
      *(_DWORD *)(v6 + 24) = 4;
      *(_QWORD *)(v6 + 32) = 0;
      *(_QWORD *)(v6 + 40) = 0;
      *(_QWORD *)(v6 + 48) = 0;
      *(_QWORD *)(v6 + 64) = 1;
      *(_QWORD *)(v6 + 72) = 0;
      *(_QWORD *)(v6 + 80) = 0;
      *(_QWORD *)(v6 + 96) = 0;
      *(_QWORD *)(v6 + 104) = 0;
      *(_QWORD *)(v6 + 120) = 1;
      *(_QWORD *)(v6 + 128) = 0;
      *(_QWORD *)(v6 + 136) = 0;
      *(_QWORD *)(v6 + 152) = 0;
      *(_QWORD *)(v6 + 160) = 0;
      *(_BYTE *)(v6 + 168) = 0;
      *(_QWORD *)(v6 + 184) = 0;
      *(_QWORD *)(v6 + 416) = v6 + 432;
      *(_DWORD *)(v6 + 560) = 0;
      *(_QWORD *)(v6 + 384) = 1;
      *(_QWORD *)(v6 + 392) = 0;
      *(_QWORD *)(v6 + 400) = 0;
      *(_DWORD *)(v6 + 408) = 0;
      *(_OWORD *)(v6 + 336) = 0;
      *(_QWORD *)v6 = off_4A31E10;
      *(_QWORD *)(v6 + 176) = &unk_4A31EC8;
      *(_OWORD *)(v6 + 352) = 0;
      *(_OWORD *)(v6 + 368) = 0;
      v10 = *(_QWORD *)(v4 + 8);
      v4 = v6 + 176;
    }
    else
    {
      v10 = *(_QWORD *)(v4 + 8);
      v9 = 0;
      v4 = 176;
    }
    v11 = *(unsigned int *)(v10 + 120);
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 124) )
    {
      v13 = v9;
      sub_C8D5F0(v10 + 112, (const void *)(v10 + 128), v11 + 1, 8u, v9, v7);
      v11 = *(unsigned int *)(v10 + 120);
      v9 = v13;
    }
    v12 = v9;
    *(_QWORD *)(*(_QWORD *)(v10 + 112) + 8 * v11) = v9;
    ++*(_DWORD *)(v10 + 120);
    sub_B8B080(v10, v8);
    sub_B841D0((char **)a2, v12);
  }
  return sub_B88F40(v4, a1, 1);
}
