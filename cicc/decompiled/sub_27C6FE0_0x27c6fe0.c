// Function: sub_27C6FE0
// Address: 0x27c6fe0
//
void __fastcall sub_27C6FE0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  int v12; // eax
  _QWORD *v13; // rdi
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // rdi
  int v16; // r13d
  unsigned __int64 v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == a2 + 48 )
    goto LABEL_19;
  if ( !v4 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
  {
LABEL_19:
    sub_27C1150(a1, a2, a3);
    BUG();
  }
  v6 = sub_27C1150(a1, a2, a3);
  v8 = *(_QWORD *)(v4 - 120);
  if ( v8 )
  {
    v9 = *(_QWORD *)(v4 - 112);
    **(_QWORD **)(v4 - 104) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v4 - 104);
  }
  *(_QWORD *)(v4 - 120) = v6;
  if ( v6 )
  {
    v10 = *(_QWORD *)(v6 + 16);
    *(_QWORD *)(v4 - 112) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v4 - 112;
    *(_QWORD *)(v4 - 104) = v6 + 16;
    *(_QWORD *)(v6 + 16) = v4 - 120;
  }
  if ( !*(_QWORD *)(v8 + 16) )
  {
    v11 = *(unsigned int *)(a4 + 8);
    v12 = v11;
    if ( *(_DWORD *)(a4 + 12) <= (unsigned int)v11 )
    {
      v14 = (unsigned __int64 *)sub_C8D7D0(a4, a4 + 16, 0, 0x18u, v17, v7);
      v15 = &v14[3 * *(unsigned int *)(a4 + 8)];
      if ( v15 )
      {
        *v15 = 6;
        v15[1] = 0;
        v15[2] = v8;
        if ( v8 != -8192 && v8 != -4096 )
          sub_BD73F0((__int64)v15);
      }
      sub_F17F80(a4, v14);
      v16 = v17[0];
      if ( a4 + 16 != *(_QWORD *)a4 )
        _libc_free(*(_QWORD *)a4);
      ++*(_DWORD *)(a4 + 8);
      *(_QWORD *)a4 = v14;
      *(_DWORD *)(a4 + 12) = v16;
    }
    else
    {
      v13 = (_QWORD *)(*(_QWORD *)a4 + 24 * v11);
      if ( v13 )
      {
        *v13 = 6;
        v13[1] = 0;
        v13[2] = v8;
        if ( v8 != -8192 && v8 != -4096 )
          sub_BD73F0((__int64)v13);
        v12 = *(_DWORD *)(a4 + 8);
      }
      *(_DWORD *)(a4 + 8) = v12 + 1;
    }
  }
}
