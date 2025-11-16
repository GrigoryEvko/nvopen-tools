// Function: sub_3058D00
// Address: 0x3058d00
//
__int64 __fastcall sub_3058D00(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rbx
  int v9; // r12d
  __int64 (*v10)(void); // rax
  int v11; // esi

  v5 = *(_QWORD *)(a2 + 32);
  v6 = v5 + 40LL * a4;
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 32LL);
  v8 = 40LL * (a4 + 1);
  v9 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 48) + 8LL)
                 + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(v7 + 48) + 32LL) + *(_DWORD *)(v6 + 24)))
     + *(_QWORD *)(v5 + v8 + 24);
  v10 = *(__int64 (**)(void))(*(_QWORD *)a1 + 680LL);
  if ( (char *)v10 == (char *)sub_30589C0 )
    v11 = 79 - ((*(_BYTE *)(*(_QWORD *)(v7 + 8) + 1264LL) == 0) - 1);
  else
    v11 = v10();
  sub_2EAB560((char *)v6, v11, 0, 0, 0, 0, 0, 0);
  sub_2EAB3B0(*(_QWORD *)(a2 + 32) + v8, v9, 0);
  return 0;
}
