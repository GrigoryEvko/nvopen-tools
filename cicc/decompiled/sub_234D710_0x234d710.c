// Function: sub_234D710
// Address: 0x234d710
//
__int64 __fastcall sub_234D710(__int64 a1, __int64 *a2, char a3, char a4)
{
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r14
  __int64 v13; // r15
  unsigned int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v18; // r15
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+20h] [rbp-50h]
  __int64 v26; // [rsp+30h] [rbp-40h]
  __int64 v27; // [rsp+30h] [rbp-40h]
  unsigned int v28; // [rsp+38h] [rbp-38h]
  __int64 v29; // [rsp+38h] [rbp-38h]

  v7 = *a2;
  v8 = a2[1];
  v9 = a2[2];
  v10 = a2[3];
  v11 = a2[4];
  v12 = a2[7];
  a2[7] = 0;
  v13 = a2[8];
  v14 = *((_DWORD *)a2 + 18);
  a2[8] = 0;
  ++a2[6];
  v15 = a2[5];
  *((_DWORD *)a2 + 18) = 0;
  v28 = v14;
  v22 = v7;
  v23 = v8;
  v24 = v9;
  v25 = v10;
  v26 = v15;
  v16 = sub_22077B0(0x58u);
  if ( v16 )
  {
    *(_QWORD *)(v16 + 56) = 1;
    *(_QWORD *)(v16 + 72) = v13;
    *(_QWORD *)(v16 + 48) = v26;
    *(_QWORD *)v16 = &unk_4A10178;
    *(_QWORD *)(v16 + 8) = v22;
    *(_QWORD *)(v16 + 16) = v23;
    *(_QWORD *)(v16 + 24) = v24;
    *(_QWORD *)(v16 + 32) = v25;
    *(_QWORD *)(v16 + 40) = v11;
    *(_DWORD *)(v16 + 80) = v28;
    *(_QWORD *)a1 = v16;
    *(_BYTE *)(a1 + 8) = a3;
    *(_BYTE *)(a1 + 9) = a4;
    v27 = 0;
    *(_QWORD *)(v16 + 64) = v12;
    v12 = 0;
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_BYTE *)(a1 + 8) = a3;
    *(_BYTE *)(a1 + 9) = a4;
    v27 = 72LL * v28;
    if ( v28 )
    {
      v18 = v12;
      v29 = v12 + 72LL * v28;
      do
      {
        if ( *(_QWORD *)v18 != -8192 && *(_QWORD *)v18 != -4096 )
        {
          v19 = *(_QWORD **)(v18 + 8);
          v20 = &v19[3 * *(unsigned int *)(v18 + 16)];
          if ( v19 != v20 )
          {
            do
            {
              v21 = *(v20 - 1);
              v20 -= 3;
              if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
                sub_BD60C0(v20);
            }
            while ( v19 != v20 );
            v20 = *(_QWORD **)(v18 + 8);
          }
          if ( v20 != (_QWORD *)(v18 + 24) )
            _libc_free((unsigned __int64)v20);
        }
        v18 += 72;
      }
      while ( v29 != v18 );
    }
  }
  sub_C7D6A0(v12, v27, 8);
  return a1;
}
