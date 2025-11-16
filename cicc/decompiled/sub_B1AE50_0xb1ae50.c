// Function: sub_B1AE50
// Address: 0xb1ae50
//
void __fastcall sub_B1AE50(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rax
  int v9; // r8d
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1[1];
  if ( v2 != a2 )
  {
    v3 = *(unsigned int *)(v2 + 32);
    v10[0] = (__int64)a1;
    v6 = *(_QWORD **)(v2 + 24);
    v7 = (__int64)&v6[v3];
    v8 = sub_B186E0(v6, v7, v10);
    if ( v8 + 1 != (_QWORD *)v7 )
    {
      memmove(v8, v8 + 1, v7 - (_QWORD)(v8 + 1));
      v9 = *(_DWORD *)(v2 + 32);
    }
    *(_DWORD *)(v2 + 32) = v9 - 1;
    a1[1] = a2;
    sub_B1AE00(a2 + 24, (__int64)a1);
    sub_B19190((__int64)a1, a1);
  }
}
