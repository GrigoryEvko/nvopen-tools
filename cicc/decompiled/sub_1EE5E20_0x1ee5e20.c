// Function: sub_1EE5E20
// Address: 0x1ee5e20
//
void __fastcall sub_1EE5E20(__int64 a1, int a2, int a3, int a4)
{
  __int64 v4; // r13
  _QWORD *v5; // rbx
  __int64 (*v6)(void); // rax
  __int64 v7; // rax
  _DWORD *v8; // rbx
  int v9; // ecx
  __int64 v10; // rdx
  unsigned __int64 v11; // r14

  if ( !a4 && a3 )
  {
    v4 = 0;
    v5 = *(_QWORD **)(a1 + 24);
    v6 = *(__int64 (**)(void))(**(_QWORD **)(*v5 + 16LL) + 112LL);
    if ( v6 != sub_1D00B10 )
      v4 = v6();
    v7 = *(_QWORD *)v4;
    if ( a2 < 0 )
    {
      v11 = *(_QWORD *)(v5[3] + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
      v8 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(v7 + 224))(v4, v11);
      v9 = *(_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v4 + 184LL))(v4, v11);
    }
    else
    {
      v8 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(v7 + 232))(v4, (unsigned int)a2);
      v9 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 192LL))(v4, (unsigned int)a2);
    }
    if ( *v8 == -1 )
      v8 = 0;
    do
    {
      if ( !v8 )
        break;
      v10 = (unsigned int)*v8++;
      *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4 * v10) -= v9;
    }
    while ( *v8 != -1 );
  }
}
