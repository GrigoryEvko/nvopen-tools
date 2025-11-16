// Function: sub_9C34D0
// Address: 0x9c34d0
//
__int64 *__fastcall sub_9C34D0(__int64 *a1, int a2, int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  int v8; // eax

  v6 = sub_22077B0(72);
  v7 = v6;
  if ( v6 )
  {
    *(_DWORD *)(v6 + 8) = 2;
    *(_DWORD *)(v6 + 12) = a2;
    *(_QWORD *)(v6 + 16) = 0;
    *(_QWORD *)v6 = &unk_49D9770;
    *(_QWORD *)(v6 + 40) = v6 + 56;
    v8 = *(_DWORD *)(a4 + 8);
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    if ( v8 )
      sub_9C2F00(v7 + 40, (char **)a4);
    *(_QWORD *)(v7 + 56) = 0;
    *(_DWORD *)(v7 + 64) = a3;
    *(_QWORD *)v7 = &unk_49D97D0;
  }
  *a1 = v7;
  return a1;
}
