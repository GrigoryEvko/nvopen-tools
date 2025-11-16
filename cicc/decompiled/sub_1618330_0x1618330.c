// Function: sub_1618330
// Address: 0x1618330
//
void __fastcall sub_1618330(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  bool v5; // zf
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax

  v4 = *(_QWORD *)(a2 + 8);
  if ( *(_QWORD *)a2 != v4
    && (v5 = (*(unsigned int (__fastcall **)(_QWORD))(**(_QWORD **)(v4 - 8) + 40LL))(*(_QWORD *)(v4 - 8)) == 6,
        v4 = *(_QWORD *)(a2 + 8),
        v5) )
  {
    v8 = *(_QWORD *)(v4 - 8);
  }
  else
  {
    v6 = *(_QWORD *)(v4 - 8);
    v7 = sub_22077B0(568);
    v8 = v7;
    if ( v7 )
    {
      *(_QWORD *)(v7 + 8) = 0;
      v9 = v7 + 40;
      *(_QWORD *)(v9 - 24) = 0;
      *(_QWORD *)(v8 + 24) = v9;
      *(_QWORD *)(v8 + 32) = 0x1000000000LL;
      *(_QWORD *)(v8 + 264) = 0x1000000000LL;
      *(_QWORD *)(v8 + 424) = &unk_4F9E3BB;
      *(_QWORD *)(v8 + 488) = v8 + 472;
      *(_QWORD *)(v8 + 496) = v8 + 472;
      *(_QWORD *)(v8 + 536) = v8 + 520;
      *(_QWORD *)(v8 + 256) = v8 + 272;
      *(_DWORD *)(v8 + 400) = 0;
      *(_QWORD *)(v8 + 168) = 0;
      *(_QWORD *)(v8 + 176) = 0;
      *(_QWORD *)(v8 + 184) = 0;
      *(_QWORD *)(v8 + 192) = 0;
      *(_QWORD *)(v8 + 200) = 0;
      *(_QWORD *)(v8 + 208) = 0;
      *(_QWORD *)(v8 + 216) = 0;
      *(_QWORD *)(v8 + 224) = 1;
      *(_QWORD *)(v8 + 232) = 0;
      *(_QWORD *)(v8 + 240) = 0;
      *(_DWORD *)(v8 + 248) = 0;
      *(_QWORD *)(v8 + 416) = 0;
      *(_DWORD *)(v8 + 432) = 3;
      *(_QWORD *)(v8 + 440) = 0;
      *(_QWORD *)(v8 + 448) = 0;
      *(_QWORD *)(v8 + 456) = 0;
      *(_DWORD *)(v8 + 472) = 0;
      *(_QWORD *)(v8 + 480) = 0;
      *(_QWORD *)(v8 + 504) = 0;
      *(_DWORD *)(v8 + 520) = 0;
      *(_QWORD *)(v8 + 528) = 0;
      *(_QWORD *)(v8 + 544) = v8 + 520;
      *(_QWORD *)v8 = off_49EDAA8;
      *(_QWORD *)(v8 + 552) = 0;
      *(_BYTE *)(v8 + 560) = 0;
      *(_QWORD *)(v8 + 408) = &unk_49EDB20;
    }
    v10 = *(_QWORD *)(v6 + 16);
    v11 = *(unsigned int *)(v10 + 120);
    if ( (unsigned int)v11 >= *(_DWORD *)(v10 + 124) )
    {
      sub_16CD150(v10 + 112, v10 + 128, 0, 8);
      v11 = *(unsigned int *)(v10 + 120);
    }
    *(_QWORD *)(*(_QWORD *)(v10 + 112) + 8 * v11) = v8;
    ++*(_DWORD *)(v10 + 120);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)(v8 + 408) + 64LL))(v8 + 408, a2, a3);
    sub_16110B0((char **)a2, v8);
  }
  sub_1617B20(v8, a1, 1);
}
