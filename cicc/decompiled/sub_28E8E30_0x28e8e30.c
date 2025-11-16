// Function: sub_28E8E30
// Address: 0x28e8e30
//
_QWORD *__fastcall sub_28E8E30(_QWORD *a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  char v6; // al
  _QWORD *v7; // rsi
  _QWORD *v8; // rdx

  v5 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v6 = sub_28E8D40(a2, a3, (__int64 *)(v5 + 8));
  v7 = a1 + 4;
  v8 = a1 + 10;
  if ( v6 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v7;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v8;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v7;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v8;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
