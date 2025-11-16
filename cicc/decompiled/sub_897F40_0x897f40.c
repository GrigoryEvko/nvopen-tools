// Function: sub_897F40
// Address: 0x897f40
//
__int64 __fastcall sub_897F40(unsigned int a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v8; // r9
  __int64 v9; // r15
  _QWORD *v10; // rdi
  __int64 v11; // rax
  int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+18h] [rbp-58h]
  __int64 v26[10]; // [rsp+20h] [rbp-50h] BYREF

  v6 = a4;
  v8 = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C44 + 616);
  v9 = *(_QWORD *)v8;
  v10 = *(_QWORD **)(*(_QWORD *)v8 + 352LL);
  if ( v10 )
  {
    v24 = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C44 + 616);
    sub_869FD0(v10, dword_4F04C64);
    v6 = a4;
    v8 = v24;
    *(_QWORD *)(v9 + 352) = 0;
  }
  if ( (*(_BYTE *)(v6 + 18) & 2) != 0 )
  {
    v11 = *(_QWORD *)(v6 + 32);
    v12 = 0;
    do
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v11 + 168) + 160LL);
      if ( v13 )
        v12 += *(_BYTE *)(v13 + 120) == 1;
      if ( (*(_BYTE *)(v11 + 89) & 4) == 0 )
        break;
      v11 = *(_QWORD *)(*(_QWORD *)(v11 + 40) + 32LL);
    }
    while ( v11 );
    if ( *(_DWORD *)(v8 + 168) == v12 )
    {
      v26[0] = 0;
      v25 = v8;
      v19 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v20 = *(_QWORD *)(v19 + 512);
      *(_QWORD *)(v19 + 512) = 0;
      ++*(_QWORD *)(v8 + 224);
      v21 = v20;
      v23 = *(_QWORD *)(v8 + 192);
      sub_890230(v8, &dword_4F077C8, v26);
      v8 = v25;
      *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 512) = v21;
      *(_QWORD *)(v26[0] + 24) = v23;
      ++*(_DWORD *)(v25 + 168);
    }
  }
  sub_897CB0(v8, v9);
  sub_643F80(v9, a2);
  sub_65C040(v9);
  *(_BYTE *)(v9 + 121) &= ~0x80u;
  *(_BYTE *)(v9 + 133) |= 8u;
  sub_87E3B0(a3);
  sub_7ADF70((__int64)v26, 0);
  sub_7AE700((__int64)(qword_4F061C0 + 3), a1, dword_4F06650[0], 0, (__int64)v26);
  return sub_7BC000((unsigned __int64)v26, a1, v14, v15, v16, v17);
}
