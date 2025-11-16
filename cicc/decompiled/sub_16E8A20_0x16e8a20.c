// Function: sub_16E8A20
// Address: 0x16e8a20
//
__off_t __fastcall sub_16E8A20(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, int a5, char a6, unsigned int a7)
{
  int v8; // ecx
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // [rsp+1Ch] [rbp-44h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD *v19; // [rsp+30h] [rbp-30h] BYREF
  __int16 v20; // [rsp+40h] [rbp-20h]

  v18[0] = a2;
  v18[1] = a3;
  if ( a3 == 1 && *a2 == 45 )
  {
    v14 = sub_2241E40(a1, a2, 1, a4, a7);
    *(_DWORD *)a4 = 0;
    v12 = 1;
    *(_QWORD *)(a4 + 8) = v14;
    v10 = a7 & 1;
    if ( (a7 & 1) == 0 )
    {
      sub_16C81F0(a1, 1, v15, v16, v10);
      v12 = 1;
    }
  }
  else
  {
    v20 = 261;
    v8 = 3;
    v19 = v18;
    if ( (a6 & 1) == 0 )
      v8 = 2;
    v9 = sub_16C5A80((__int64)&v19, (int *)&v17, a5, v8, a7, 0x1B6u);
    *(_DWORD *)a4 = v9;
    *(_QWORD *)(a4 + 8) = v11;
    if ( v9 )
      v12 = 0xFFFFFFFFLL;
    else
      v12 = v17;
  }
  return sub_16E8970(a1, v12, 1, 0, v10);
}
