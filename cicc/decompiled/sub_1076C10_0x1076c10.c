// Function: sub_1076C10
// Address: 0x1076c10
//
__int64 *__fastcall sub_1076C10(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rax

  v3 = *a2;
  *a2 = 0;
  v5 = sub_22077B0(144);
  if ( v5 )
  {
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 8) = v5 + 24;
    *(_QWORD *)(v5 + 24) = v5 + 40;
    *(_WORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 88) = v5 + 104;
    *(_QWORD *)(v5 + 32) = 0;
    *(_BYTE *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_QWORD *)(v5 + 72) = 0;
    *(_QWORD *)(v5 + 96) = 0;
    *(_QWORD *)v5 = &unk_49E6108;
    *(_QWORD *)(v5 + 104) = a3;
    *(_DWORD *)(v5 + 112) = 1;
    *(_QWORD *)(v5 + 120) = v3;
    *(_QWORD *)(v5 + 128) = 0;
    *(_DWORD *)(v5 + 136) = 0;
LABEL_3:
    *a1 = v5;
    return a1;
  }
  if ( !v3 )
    goto LABEL_3;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  *a1 = 0;
  return a1;
}
