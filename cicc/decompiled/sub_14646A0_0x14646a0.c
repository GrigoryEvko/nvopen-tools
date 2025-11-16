// Function: sub_14646A0
// Address: 0x14646a0
//
__int64 __fastcall sub_14646A0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // r14
  __int64 v5; // r14
  char v7; // r15
  int v8; // edx
  unsigned int v9; // eax
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // r8
  int v13; // r9d
  void *v14; // [rsp+0h] [rbp-90h] BYREF
  _BYTE v15[16]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v16; // [rsp+18h] [rbp-78h]
  void *v17; // [rsp+30h] [rbp-60h] BYREF
  _BYTE v18[16]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v19; // [rsp+48h] [rbp-48h]

  v2 = *(_DWORD *)(a1 + 168);
  if ( v2 )
  {
    v5 = *(_QWORD *)(a1 + 152);
    v7 = 1;
    sub_1457D90(&v14, -8, 0);
    sub_1457D90(&v17, -16, 0);
    v8 = v2 - 1;
    v9 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = v5 + 48LL * v9;
    v11 = *(_QWORD *)(v10 + 24);
    if ( v11 != a2 )
    {
      v12 = v5 + 48LL * (v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)));
      v13 = 1;
      v10 = 0;
      while ( v16 != v11 )
      {
        if ( v19 != v11 || v10 )
          v12 = v10;
        v9 = v8 & (v13 + v9);
        v10 = v5 + 48LL * v9;
        v11 = *(_QWORD *)(v10 + 24);
        if ( v11 == a2 )
        {
          v7 = 1;
          goto LABEL_5;
        }
        ++v13;
        v10 = v12;
        v12 = v5 + 48LL * v9;
      }
      v7 = 0;
      if ( !v10 )
        v10 = v12;
    }
LABEL_5:
    v17 = &unk_49EE2B0;
    if ( v19 != -8 && v19 != 0 && v19 != -16 )
      sub_1649B30(v18);
    v14 = &unk_49EE2B0;
    if ( v16 != 0 && v16 != -8 && v16 != -16 )
      sub_1649B30(v15);
    if ( v7 && v10 != *(_QWORD *)(a1 + 152) + 48LL * *(unsigned int *)(a1 + 168) )
    {
      v3 = *(_QWORD *)(v10 + 40);
      if ( (unsigned __int8)sub_145A670(a1, v3) )
        return v3;
      sub_1464220(a1, a2);
      sub_1459590(a1, v3);
    }
  }
  return 0;
}
