// Function: sub_104A830
// Address: 0x104a830
//
void __fastcall sub_104A830(_BYTE *a1, __int64 a2)
{
  _QWORD *v3; // r12
  _QWORD *v4; // rdi
  __int64 v5; // rsi
  _QWORD *v6; // rax
  int v7; // r9d
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _BYTE *v10; // [rsp-30h] [rbp-30h] BYREF

  if ( *a1 > 0x1Cu )
  {
    v3 = a1;
    v10 = a1;
    v4 = *(_QWORD **)a2;
    v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
    v6 = sub_104A710(v4, v5, (__int64 *)&v10);
    if ( (_QWORD *)v5 == v6 )
    {
      v8 = 32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF);
      if ( (*((_BYTE *)v3 + 7) & 0x40) != 0 )
      {
        v9 = (_QWORD *)*(v3 - 1);
        v3 = &v9[(unsigned __int64)v8 / 8];
      }
      else
      {
        v9 = &v3[v8 / 0xFFFFFFFFFFFFFFF8LL];
      }
      for ( ; v3 != v9; v9 += 4 )
      {
        if ( *(_BYTE *)*v9 > 0x1Cu )
          sub_104A830(*v9, a2);
      }
    }
    else
    {
      if ( (_QWORD *)v5 != v6 + 1 )
      {
        memmove(v6, v6 + 1, v5 - (_QWORD)(v6 + 1));
        v7 = *(_DWORD *)(a2 + 8);
      }
      *(_DWORD *)(a2 + 8) = v7 - 1;
    }
  }
}
