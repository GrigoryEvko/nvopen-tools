// Function: sub_3915AB0
// Address: 0x3915ab0
//
__int64 *__fastcall sub_3915AB0(__int64 *a1, __int64 *a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rbx

  v6 = *a2;
  *a2 = 0;
  v7 = sub_22077B0(0x100u);
  v8 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 8) = v6;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)v7 = &unk_4A3EBF8;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_DWORD *)(v7 + 40) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 56) = 0;
    *(_QWORD *)(v7 + 64) = 0;
    *(_DWORD *)(v7 + 72) = 0;
    *(_QWORD *)(v7 + 80) = 0;
    *(_QWORD *)(v7 + 88) = 0;
    *(_QWORD *)(v7 + 96) = 0;
    *(_DWORD *)(v7 + 104) = 0;
    sub_167FAB0(v7 + 112, 2, 1);
    *(_QWORD *)(v8 + 240) = a3;
    *(_QWORD *)(v8 + 168) = 0;
    *(_QWORD *)(v8 + 176) = 0;
    *(_QWORD *)(v8 + 184) = 0;
    *(_QWORD *)(v8 + 192) = 0;
    *(_QWORD *)(v8 + 200) = 0;
    *(_QWORD *)(v8 + 208) = 0;
    *(_QWORD *)(v8 + 216) = 0;
    *(_QWORD *)(v8 + 224) = 0;
    *(_QWORD *)(v8 + 232) = 0;
    *(_DWORD *)(v8 + 248) = a4;
  }
  else if ( v6 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  }
  *a1 = v8;
  return a1;
}
