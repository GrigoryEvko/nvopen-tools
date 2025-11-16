// Function: sub_384D0D0
// Address: 0x384d0d0
//
void __fastcall sub_384D0D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // eax
  __int64 v4; // rbx
  __int64 v5; // rax
  int v6; // r9d
  __int64 *v7; // r13
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  if ( v2 != *(_QWORD *)a2 )
  {
    while ( (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(v2 - 8) + 40LL))(*(_QWORD *)(v2 - 8)) > 2 )
    {
      sub_160FB80(a2);
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
      v4 = 160;
  }
  else
  {
    v5 = sub_22077B0(0x238u);
    v7 = (__int64 *)v5;
    if ( v5 )
    {
      *(_QWORD *)(v5 + 8) = 0;
      *(_QWORD *)(v5 + 16) = &unk_50516DE;
      v8 = v5 + 160;
      *(_QWORD *)(v5 + 80) = v5 + 64;
      *(_QWORD *)(v5 + 88) = v5 + 64;
      *(_QWORD *)(v5 + 128) = v5 + 112;
      *(_QWORD *)(v5 + 136) = v5 + 112;
      *(_QWORD *)(v5 + 184) = v5 + 200;
      *(_QWORD *)(v5 + 192) = 0x1000000000LL;
      *(_QWORD *)(v5 + 424) = 0x1000000000LL;
      *(_DWORD *)(v5 + 24) = 5;
      *(_QWORD *)(v5 + 32) = 0;
      *(_QWORD *)(v5 + 40) = 0;
      *(_QWORD *)(v5 + 48) = 0;
      *(_DWORD *)(v5 + 64) = 0;
      *(_QWORD *)(v5 + 72) = 0;
      *(_QWORD *)(v5 + 96) = 0;
      *(_DWORD *)(v5 + 112) = 0;
      *(_QWORD *)(v5 + 120) = 0;
      *(_QWORD *)(v5 + 144) = 0;
      *(_BYTE *)(v5 + 152) = 0;
      *(_QWORD *)(v5 + 168) = 0;
      *(_QWORD *)(v5 + 176) = 0;
      *(_QWORD *)(v5 + 416) = v5 + 432;
      *(_DWORD *)(v5 + 560) = 0;
      *(_QWORD *)(v5 + 328) = 0;
      *(_QWORD *)(v5 + 336) = 0;
      *(_QWORD *)(v5 + 344) = 0;
      *(_QWORD *)(v5 + 352) = 0;
      *(_QWORD *)(v5 + 360) = 0;
      *(_QWORD *)(v5 + 368) = 0;
      *(_QWORD *)(v5 + 376) = 0;
      *(_QWORD *)(v5 + 384) = 1;
      *(_QWORD *)(v5 + 392) = 0;
      *(_QWORD *)v5 = off_4A3DBA8;
      *(_QWORD *)(v5 + 400) = 0;
      *(_DWORD *)(v5 + 408) = 0;
      *(_QWORD *)(v5 + 160) = &unk_4A3DC60;
      v9 = *(_QWORD *)(v4 + 16);
      v4 = v5 + 160;
    }
    else
    {
      v9 = *(_QWORD *)(v4 + 16);
      v8 = 0;
      v4 = 160;
    }
    v10 = *(unsigned int *)(v9 + 120);
    if ( (unsigned int)v10 >= *(_DWORD *)(v9 + 124) )
    {
      v12 = v8;
      sub_16CD150(v9 + 112, (const void *)(v9 + 128), 0, 8, v8, v6);
      v10 = *(unsigned int *)(v9 + 120);
      v8 = v12;
    }
    v11 = v8;
    *(_QWORD *)(*(_QWORD *)(v9 + 112) + 8 * v10) = v8;
    ++*(_DWORD *)(v9 + 120);
    sub_16185C0(v9, v7);
    sub_16110B0((char **)a2, v11);
  }
  sub_1617B20(v4, a1, 1);
}
