// Function: sub_30590F0
// Address: 0x30590f0
//
void __fastcall sub_30590F0(__int64 a1, int a2, __int64 a3, __int64 a4, int a5, unsigned int a6)
{
  unsigned __int64 *v7; // r13
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 *v10; // r14
  unsigned __int64 *v11; // r12
  __int128 v12; // [rsp-28h] [rbp-48h]

  *((_QWORD *)&v12 + 1) = -1;
  *(_QWORD *)&v12 = -1;
  sub_2FF6100(
    a1,
    (__int64)&off_49D6D30,
    (__int64)&unk_4A2F8A8 - 104,
    (__int64)&unk_4A2F8A8,
    (__int64)&off_4CE0070,
    (__int64)&unk_44C7FA0,
    (__int64)&unk_44C7F90,
    v12,
    (__int64)&unk_44C7EC0,
    (__int64)&unk_44C7FC0,
    a6);
  *(_DWORD *)(a1 + 20) = a2;
  v7 = *(unsigned __int64 **)(a1 + 232);
  *(_DWORD *)(a1 + 16) = 103;
  *(_DWORD *)(a1 + 24) = a5;
  *(_QWORD *)a1 = &unk_4A2F290;
  *(_DWORD *)(a1 + 96) = 1;
  *(_QWORD *)(a1 + 8) = &unk_4458100;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 32) = &off_49D44C0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 56) = &unk_4458D10;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 64) = &unk_4458D00;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 72) = "ENVREG10";
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 80) = "Int1Regs";
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 48) = &unk_4457F60;
  *(_QWORD *)(a1 + 40) = 0x660000000DLL;
  *(_QWORD *)(a1 + 88) = &unk_4458CF2;
  *(_QWORD *)(a1 + 104) = &unk_4457C40;
  v8 = *(_QWORD *)(a1 + 224);
  v9 = (__int64)v7 - v8;
  if ( (unsigned __int64)v7 - v8 <= 0x990 )
  {
    sub_301EE80(a1 + 224, 103 - 0xAAAAAAAAAAAAAAABLL * (v9 >> 3));
  }
  else if ( (unsigned __int64)v9 > 0x9A8 )
  {
    v10 = (unsigned __int64 *)(v8 + 2472);
    if ( v7 != v10 )
    {
      v11 = v10;
      do
      {
        if ( *v11 )
          j_j___libc_free_0(*v11);
        v11 += 3;
      }
      while ( v7 != v11 );
      *(_QWORD *)(a1 + 232) = v10;
    }
  }
}
