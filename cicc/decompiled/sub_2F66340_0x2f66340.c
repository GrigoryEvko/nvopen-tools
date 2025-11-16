// Function: sub_2F66340
// Address: 0x2f66340
//
bool __fastcall sub_2F66340(_DWORD *a1, __int64 a2)
{
  bool result; // al
  __int16 v4; // ax
  _DWORD *v5; // rdx
  unsigned int v6; // r15d
  unsigned int v7; // ebx
  unsigned int v8; // r14d
  unsigned int v9; // r12d
  int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // esi
  __int64 v13; // rsi
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned int v19; // eax
  unsigned int v20; // eax

  result = 0;
  if ( a2 )
  {
    v4 = *(_WORD *)(a2 + 68);
    if ( v4 == 20 )
    {
      v5 = *(_DWORD **)(a2 + 32);
      v6 = v5[2];
      v7 = v5[12];
      v8 = (*v5 >> 8) & 0xFFF;
      v9 = (v5[10] >> 8) & 0xFFF;
    }
    else
    {
      if ( v4 != 12 )
        return 0;
      v17 = *(_QWORD *)(a2 + 32);
      v18 = *(_QWORD *)(v17 + 144);
      v6 = *(_DWORD *)(v17 + 8);
      v8 = v18;
      if ( ((*(_DWORD *)v17 >> 8) & 0xFFF) != 0 )
      {
        v8 = (*(_DWORD *)v17 >> 8) & 0xFFF;
        if ( (_DWORD)v18 )
        {
          v19 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)a1 + 296LL))(
                  *(_QWORD *)a1,
                  (*(_DWORD *)v17 >> 8) & 0xFFF);
          v17 = *(_QWORD *)(a2 + 32);
          v8 = v19;
        }
      }
      v7 = *(_DWORD *)(v17 + 88);
      v9 = (*(_DWORD *)(v17 + 80) >> 8) & 0xFFF;
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
    if ( v12 - 1 > 0x3FFFFFFE )
    {
      if ( v12 != v7 )
        return 0;
      v13 = (unsigned int)a1[5];
      v14 = *(_QWORD *)a1;
      if ( (_DWORD)v13 )
      {
        if ( v8 )
        {
          v15 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v14 + 296LL))(v14, v13, v8);
          v14 = *(_QWORD *)a1;
          v8 = v15;
        }
        else
        {
          v8 = v13;
        }
      }
      v16 = (unsigned int)a1[4];
      if ( (_DWORD)v16 )
      {
        if ( v9 )
          v9 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v14 + 296LL))(v14, v16, v9);
        else
          v9 = a1[4];
      }
      return v9 == v8;
    }
    else
    {
      if ( v7 - 1 > 0x3FFFFFFE )
        return 0;
      if ( v9 )
      {
        v20 = sub_E91CF0(*(_QWORD **)a1, v7, v9);
        v12 = a1[2];
        v7 = v20;
      }
      if ( v8 )
        return (unsigned int)sub_E91CF0(*(_QWORD **)a1, v12, v8) == v7;
      else
        return v12 == v7;
    }
  }
  return result;
}
