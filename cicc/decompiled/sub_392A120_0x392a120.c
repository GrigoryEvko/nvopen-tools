// Function: sub_392A120
// Address: 0x392a120
//
__int64 *__fastcall sub_392A120(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rax

  v4 = *a2;
  *a2 = 0;
  v5 = sub_22077B0(0xE8u);
  if ( v5 )
  {
    *(_QWORD *)v5 = off_4A3ECB8;
    *(_QWORD *)(v5 + 8) = a3;
    *(_DWORD *)(v5 + 16) = 1;
    *(_QWORD *)(v5 + 24) = v4;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_QWORD *)(v5 + 72) = 0;
    *(_QWORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 88) = 0;
    *(_QWORD *)(v5 + 96) = 0;
    *(_OWORD *)(v5 + 32) = 0;
    sub_167FAB0(v5 + 104, 1, 1);
    v6 = *(_QWORD *)(v5 + 24);
    *(_QWORD *)(v5 + 160) = 0;
    *(_QWORD *)(v5 + 168) = 0;
    *(_QWORD *)(v5 + 176) = 0;
    *(_DWORD *)(v5 + 184) = 0;
    *(_QWORD *)(v5 + 192) = 0;
    *(_QWORD *)(v5 + 200) = 0;
    *(_QWORD *)(v5 + 208) = 0;
    *(_DWORD *)(v5 + 216) = 0;
    *(_WORD *)(v5 + 32) = *(_DWORD *)(v6 + 8);
LABEL_3:
    *a1 = v5;
    return a1;
  }
  if ( !v4 )
    goto LABEL_3;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  *a1 = 0;
  return a1;
}
