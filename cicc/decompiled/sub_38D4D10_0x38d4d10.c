// Function: sub_38D4D10
// Address: 0x38d4d10
//
void __fastcall sub_38D4D10(__int64 a1, const void *a2, size_t a3)
{
  __int64 v5; // rax
  unsigned int *v6; // rsi
  __int64 v7; // rbx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rdi

  sub_39120A0();
  v5 = *(unsigned int *)(a1 + 120);
  v6 = 0;
  if ( (_DWORD)v5 )
    v6 = *(unsigned int **)(*(_QWORD *)(a1 + 112) + 32 * v5 - 32);
  sub_38CB070((_QWORD *)a1, v6);
  v7 = sub_38D4BB0(a1, 0);
  sub_38D4150(a1, v7, *(unsigned int *)(v7 + 72));
  v10 = *(unsigned int *)(v7 + 72);
  if ( a3 > (unsigned __int64)*(unsigned int *)(v7 + 76) - v10 )
  {
    sub_16CD150(v7 + 64, (const void *)(v7 + 80), a3 + v10, 1, v8, v9);
    v10 = *(unsigned int *)(v7 + 72);
  }
  if ( a3 )
  {
    memcpy((void *)(*(_QWORD *)(v7 + 64) + v10), a2, a3);
    LODWORD(v10) = *(_DWORD *)(v7 + 72);
  }
  *(_DWORD *)(v7 + 72) = v10 + a3;
}
