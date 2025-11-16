// Function: sub_1253350
// Address: 0x1253350
//
__int64 *__fastcall sub_1253350(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 (__fastcall *v8)(__int64); // rax

  v4 = *a2;
  *a2 = 0;
  v5 = sub_22077B0(264);
  v6 = v5;
  if ( v5 )
  {
    *(_QWORD *)(v5 + 8) = v5 + 24;
    *(_QWORD *)(v5 + 24) = v5 + 40;
    *(_WORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 88) = v5 + 104;
    *(_QWORD *)v5 = off_49E6768;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_BYTE *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_QWORD *)(v5 + 72) = 0;
    *(_QWORD *)(v5 + 96) = 0;
    *(_QWORD *)(v5 + 104) = v4;
    *(_DWORD *)(v5 + 120) = 0;
    *(_BYTE *)(v5 + 152) = 0;
    *(_DWORD *)(v5 + 156) = 1;
    *(_QWORD *)(v5 + 144) = 0;
    *(_QWORD *)(v5 + 136) = 0;
    *(_QWORD *)(v5 + 128) = 0;
    *(_QWORD *)(v5 + 112) = off_49E66F0;
    *(_QWORD *)(v5 + 160) = a3;
    *(_QWORD *)(v5 + 168) = 0;
    *(_DWORD *)(v5 + 176) = 0;
    *(_BYTE *)(v5 + 181) = 0;
    sub_CB5980(v5 + 112, v5 + 182, 77, 2);
LABEL_3:
    *a1 = v6;
    return a1;
  }
  if ( !v4 )
    goto LABEL_3;
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL);
  if ( v8 == sub_106DB80 )
    j_j___libc_free_0(v4, 8);
  else
    v8(v4);
  *a1 = 0;
  return a1;
}
