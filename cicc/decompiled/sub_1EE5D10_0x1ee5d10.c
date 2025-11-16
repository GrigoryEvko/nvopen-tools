// Function: sub_1EE5D10
// Address: 0x1ee5d10
//
void __fastcall sub_1EE5D10(__int64 a1, int a2, int a3, int a4)
{
  __int64 v4; // r13
  _QWORD *v6; // rbx
  __int64 (*v7)(void); // rax
  __int64 v8; // rax
  _DWORD *v9; // rbx
  int v10; // edi
  __int64 v11; // rdx
  __int64 v12; // rsi
  int *v13; // rcx
  int v14; // eax
  unsigned __int64 v15; // r14

  if ( !a3 && a4 )
  {
    v4 = 0;
    v6 = *(_QWORD **)(a1 + 24);
    v7 = *(__int64 (**)(void))(**(_QWORD **)(*v6 + 16LL) + 112LL);
    if ( v7 != sub_1D00B10 )
      v4 = v7();
    v8 = *(_QWORD *)v4;
    if ( a2 < 0 )
    {
      v15 = *(_QWORD *)(v6[3] + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
      v9 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(v8 + 224))(v4, v15);
      v10 = *(_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v4 + 184LL))(v4, v15);
    }
    else
    {
      v9 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(v8 + 232))(v4, (unsigned int)a2);
      v10 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 192LL))(v4, (unsigned int)a2);
    }
    if ( *v9 == -1 )
      v9 = 0;
    do
    {
      if ( !v9 )
        break;
      *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * (unsigned int)*v9) += v10;
      v11 = (unsigned int)*v9;
      v12 = *(_QWORD *)(a1 + 72);
      v13 = (int *)(**(_QWORD **)(a1 + 48) + 4 * v11);
      v14 = *v13;
      if ( *(_DWORD *)(v12 + 4 * v11) >= (unsigned int)*v13 )
        v14 = *(_DWORD *)(v12 + 4 * v11);
      ++v9;
      *v13 = v14;
    }
    while ( *v9 != -1 );
  }
}
