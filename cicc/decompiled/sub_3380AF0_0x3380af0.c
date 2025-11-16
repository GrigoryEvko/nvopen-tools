// Function: sub_3380AF0
// Address: 0x3380af0
//
void __fastcall sub_3380AF0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rsi
  __int64 *v7; // r15
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  int v13; // edx
  _QWORD *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // [rsp+18h] [rbp-48h] BYREF
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  int v18; // [rsp+28h] [rbp-38h]

  v3 = a1[108];
  v4 = *a1;
  v17 = 0;
  v5 = *(_QWORD *)(v3 + 16);
  v18 = *((_DWORD *)a1 + 212);
  if ( v4 )
  {
    if ( &v17 != (__int64 *)(v4 + 48) )
    {
      v6 = *(_QWORD *)(v4 + 48);
      v17 = v6;
      if ( v6 )
      {
        sub_B96E90((__int64)&v17, v6, 1);
        v3 = a1[108];
      }
    }
  }
  v7 = *(__int64 **)(a2 + 8);
  v8 = sub_2E79000(*(__int64 **)(v3 + 40));
  v9 = sub_2D5BAE0(v5, v8, v7, 0);
  v11 = sub_3402A00(a1[108], &v17, v9, v10);
  v16 = a2;
  v12 = v11;
  LODWORD(v7) = v13;
  v14 = sub_337DC20((__int64)(a1 + 1), &v16);
  *v14 = v12;
  v15 = v17;
  *((_DWORD *)v14 + 2) = (_DWORD)v7;
  if ( v15 )
    sub_B91220((__int64)&v17, v15);
}
