// Function: sub_1EDB0A0
// Address: 0x1edb0a0
//
bool __fastcall sub_1EDB0A0(unsigned int *a1, __int64 a2)
{
  __int16 v3; // ax
  _DWORD *v5; // rdx
  int v6; // r15d
  int v7; // ebx
  unsigned int v8; // r14d
  unsigned int v9; // r12d
  unsigned int v10; // eax
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rsi
  int v20; // eax

  if ( !a2 )
    return 0;
  v3 = **(_WORD **)(a2 + 16);
  if ( v3 == 15 )
  {
    v5 = *(_DWORD **)(a2 + 32);
    v6 = v5[2];
    v7 = v5[12];
    v8 = (*v5 >> 8) & 0xFFF;
    v9 = (v5[10] >> 8) & 0xFFF;
  }
  else
  {
    if ( v3 != 10 )
      return 0;
    v13 = *(_QWORD *)(a2 + 32);
    v14 = *(_QWORD *)(v13 + 144);
    v6 = *(_DWORD *)(v13 + 8);
    v8 = v14;
    if ( ((*(_DWORD *)v13 >> 8) & 0xFFF) != 0 )
    {
      v8 = (*(_DWORD *)v13 >> 8) & 0xFFF;
      if ( (_DWORD)v14 )
      {
        v15 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)a1 + 120LL))(
                *(_QWORD *)a1,
                (*(_DWORD *)v13 >> 8) & 0xFFF);
        v13 = *(_QWORD *)(a2 + 32);
        v8 = v15;
      }
    }
    v7 = *(_DWORD *)(v13 + 88);
    v9 = (*(_DWORD *)(v13 + 80) >> 8) & 0xFFF;
  }
  v10 = a1[3];
  if ( v10 != v6 )
  {
    if ( v10 != v7 )
      return 0;
    v11 = v8;
    v7 = v6;
    v8 = v9;
    v9 = v11;
  }
  v12 = a1[2];
  if ( (int)v12 <= 0 )
  {
    if ( (_DWORD)v12 != v7 )
      return 0;
    v16 = a1[5];
    v17 = *(_QWORD *)a1;
    if ( (_DWORD)v16 )
    {
      if ( v8 )
      {
        v18 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v17 + 120LL))(v17, v16, v8);
        v17 = *(_QWORD *)a1;
        v8 = v18;
      }
      else
      {
        v8 = v16;
      }
    }
    v19 = a1[4];
    if ( (_DWORD)v19 )
    {
      if ( v9 )
        v9 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v17 + 120LL))(v17, v19, v9);
      else
        v9 = a1[4];
    }
    return v9 == v8;
  }
  else
  {
    if ( v7 <= 0 )
      return 0;
    if ( v9 )
    {
      v20 = sub_38D6F10(*(_QWORD *)a1 + 8LL, (unsigned int)v7, v9);
      v12 = a1[2];
      v7 = v20;
    }
    if ( v8 )
      return (unsigned int)sub_38D6F10(*(_QWORD *)a1 + 8LL, v12, v8) == v7;
    else
      return (_DWORD)v12 == v7;
  }
}
