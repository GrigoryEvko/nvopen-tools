// Function: sub_6F40D0
// Address: 0x6f40d0
//
void __fastcall sub_6F40D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15[3]; // [rsp+8h] [rbp-18h] BYREF

  v7 = *(_BYTE *)(a1 + 8);
  if ( v7 )
  {
    if ( v7 == 1 )
    {
      sub_6F41B0(*(_QWORD *)(a1 + 24));
    }
    else
    {
      if ( v7 != 2 )
        sub_721090(a1);
      v15[0] = sub_724DC0(a1, a2, a3, a4, a5, a6);
      sub_724C70(v15[0], 13);
      *(_QWORD *)(v15[0] + 128) = sub_72CBE0(v15[0], 13, v8, v9, v10, v11);
      v12 = v15[0];
      *(_BYTE *)(v15[0] + 176) |= 3u;
      *(_QWORD *)(v12 + 184) = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
      *(_QWORD *)(v12 + 128) = *(_QWORD *)&dword_4D03B80;
      *(_BYTE *)(a1 + 8) = 0;
      v13 = sub_6E2EF0();
      *(_QWORD *)(a1 + 32) = 0;
      v14 = v15[0];
      *(_QWORD *)(a1 + 24) = v13;
      sub_6E6A50(v14, (__int64)(v13 + 1));
      sub_724E30(v15);
    }
  }
  else
  {
    sub_6F40C0(*(_QWORD *)(a1 + 24) + 8LL, a2, a3, a4, a5, a6);
  }
}
