// Function: sub_392EBD0
// Address: 0x392ebd0
//
__int64 *__fastcall sub_392EBD0(__int64 *a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rbx

  v6 = *a2;
  *a2 = 0;
  v7 = sub_22077B0(0x80u);
  v8 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 8) = v6;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_DWORD *)(v7 + 40) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 56) = 0;
    *(_QWORD *)(v7 + 64) = 0;
    *(_DWORD *)(v7 + 72) = 0;
    *(_BYTE *)(v7 + 80) = 0;
    *(_QWORD *)(v7 + 88) = 0;
    *(_QWORD *)(v7 + 96) = 0;
    *(_QWORD *)(v7 + 104) = 0;
    *(_QWORD *)v7 = off_4A3ED50;
    *(_QWORD *)(v7 + 112) = a3;
    *(_BYTE *)(v7 + 120) = a4;
  }
  else if ( v6 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  }
  *a1 = v8;
  return a1;
}
