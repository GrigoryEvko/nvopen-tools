// Function: sub_7FCE40
// Address: 0x7fce40
//
__int64 __fastcall sub_7FCE40(__int64 a1, __int64 a2, unsigned int a3, int a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int16 v13; // r15
  __int64 v15; // [rsp-10h] [rbp-E0h]
  __int64 v16; // [rsp+0h] [rbp-D0h]
  unsigned int v18; // [rsp+8h] [rbp-C8h]
  _BYTE v19[4]; // [rsp+14h] [rbp-BCh] BYREF
  __int64 *v20; // [rsp+18h] [rbp-B8h] BYREF
  _BYTE v21[48]; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE v22[128]; // [rsp+50h] [rbp-80h] BYREF

  v20 = 0;
  sub_7F90D0(a2, (__int64)v22);
  sub_7F55E0(a1, (__int64)v22, (__int64)v21);
  if ( a4 )
    v22[19] = 1;
  v9 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v9 + 48) == 6 )
  {
    sub_8032D0(*(_QWORD *)(v9 + 56), (unsigned int)v22, 1, 0, 0, a6, (__int64)v19, 1);
    return v15;
  }
  else
  {
    if ( *(_BYTE *)(a1 + 8) <= 1u )
    {
      v16 = *(_QWORD *)(a1 + 24);
      sub_7FCD20(*(_QWORD **)(v9 + 80), *(_QWORD *)(a1 + 16), (__int64)v22, a5, a6, &v20);
      v9 = v16;
    }
    v10 = *(_QWORD *)(v9 + 8);
    v11 = *(_QWORD *)(v9 + 16);
    v12 = dword_4F07508[0];
    v13 = dword_4F07508[1];
    if ( v10 )
      *(_QWORD *)dword_4F07508 = *(_QWORD *)(v10 + 64);
    v18 = v12;
    sub_7FE6E0(v11, v22, a3, v20, a6);
    LOWORD(dword_4F07508[1]) = v13;
    dword_4F07508[0] = v18;
    return v18;
  }
}
