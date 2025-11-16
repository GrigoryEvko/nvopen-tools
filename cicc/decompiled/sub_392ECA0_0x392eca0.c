// Function: sub_392ECA0
// Address: 0x392eca0
//
__int64 *__fastcall sub_392ECA0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx

  v7 = *a2;
  *a2 = 0;
  v8 = sub_22077B0(0x88u);
  v9 = v8;
  if ( v8 )
  {
    *(_QWORD *)(v8 + 8) = v7;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 32) = 0;
    *(_DWORD *)(v8 + 40) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)(v8 + 56) = 0;
    *(_QWORD *)(v8 + 64) = 0;
    *(_DWORD *)(v8 + 72) = 0;
    *(_BYTE *)(v8 + 80) = 0;
    *(_QWORD *)(v8 + 88) = 0;
    *(_QWORD *)(v8 + 96) = 0;
    *(_QWORD *)(v8 + 104) = 0;
    *(_QWORD *)v8 = off_4A3EDB8;
    *(_QWORD *)(v8 + 112) = a3;
    *(_QWORD *)(v8 + 120) = a4;
    *(_BYTE *)(v8 + 128) = a5;
  }
  else if ( v7 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  }
  *a1 = v9;
  return a1;
}
