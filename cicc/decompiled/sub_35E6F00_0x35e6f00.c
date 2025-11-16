// Function: sub_35E6F00
// Address: 0x35e6f00
//
void __fastcall sub_35E6F00(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  void (__fastcall *v8)(__int64, _QWORD, _QWORD, unsigned __int64, _QWORD); // r13
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r8
  unsigned int v14; // ebx
  unsigned __int64 v15; // rax

  sub_35E6DC0((__int64)a1, a2, a3, a4, a5, a6);
  (*(void (__fastcall **)(_QWORD *))(*a1 + 48LL))(a1);
  (*(void (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)a1[9] + 80LL))(a1[9], a1[3]);
  v7 = a1[9];
  v8 = *(void (__fastcall **)(__int64, _QWORD, _QWORD, unsigned __int64, _QWORD))(*(_QWORD *)v7 + 96LL);
  v9 = sub_2E313E0(a1[3]);
  v10 = a1[3];
  v11 = v9;
  v12 = *(_QWORD *)(v10 + 56);
  if ( v12 == v11 )
  {
    v14 = 0;
  }
  else
  {
    v13 = 0;
    do
    {
      while ( 1 )
      {
        if ( !v12 )
          BUG();
        if ( (*(_BYTE *)v12 & 4) == 0 )
          break;
        v12 = *(_QWORD *)(v12 + 8);
        ++v13;
        if ( v11 == v12 )
          goto LABEL_7;
      }
      while ( (*(_BYTE *)(v12 + 44) & 8) != 0 )
        v12 = *(_QWORD *)(v12 + 8);
      v12 = *(_QWORD *)(v12 + 8);
      ++v13;
    }
    while ( v11 != v12 );
LABEL_7:
    v14 = v13;
  }
  v15 = sub_2E313E0(v10);
  v8(v7, a1[3], *(_QWORD *)(a1[3] + 56LL), v15, v14);
  sub_2F97F60(a1[9], *(_QWORD *)(a1[1] + 40LL), 0, 0, 0, 0);
}
