// Function: sub_72EDB0
// Address: 0x72edb0
//
__int64 __fastcall sub_72EDB0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  int v6; // edi
  int v7; // r12d
  _BYTE *v8; // rbx
  __int64 v9; // rax

  v6 = *(_DWORD *)(*(_QWORD *)(a4 + 32) + 164LL);
  v7 = dword_4F07270[0];
  if ( dword_4F07270[0] == v6 )
    v7 = 0;
  else
    sub_7296B0(v6);
  v8 = sub_727030();
  sub_729730(v7);
  v9 = *(_QWORD *)(a4 + 224);
  *((_QWORD *)v8 + 1) = a1;
  *((_QWORD *)v8 + 3) = a2;
  *(_QWORD *)v8 = v9;
  v8[16] = a3;
  *(_QWORD *)(a4 + 224) = v8;
  return a3;
}
