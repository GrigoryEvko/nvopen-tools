// Function: sub_234BA90
// Address: 0x234ba90
//
__int64 __fastcall sub_234BA90(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // r14
  unsigned int v5; // r12d
  __int64 v6; // r15
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v13; // r15
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-60h]
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 v23; // [rsp+28h] [rbp-38h]
  __int64 v24; // [rsp+28h] [rbp-38h]

  ++a2[6];
  v4 = a2[7];
  v5 = *((_DWORD *)a2 + 18);
  v6 = a2[8];
  a2[7] = 0;
  v7 = *a2;
  v8 = a2[1];
  a2[8] = 0;
  v9 = a2[2];
  v10 = a2[3];
  *((_DWORD *)a2 + 18) = 0;
  v17 = v7;
  v18 = v8;
  v19 = v9;
  v20 = v10;
  v21 = a2[4];
  v23 = a2[5];
  v11 = sub_22077B0(0x58u);
  if ( v11 )
  {
    *(_QWORD *)(v11 + 56) = 1;
    *(_QWORD *)(v11 + 72) = v6;
    *(_QWORD *)(v11 + 40) = v21;
    *(_QWORD *)v11 = &unk_4A10178;
    *(_QWORD *)(v11 + 8) = v17;
    *(_QWORD *)(v11 + 16) = v18;
    *(_QWORD *)(v11 + 24) = v19;
    *(_QWORD *)(v11 + 32) = v20;
    *(_QWORD *)(v11 + 48) = v23;
    *(_DWORD *)(v11 + 80) = v5;
    *(_QWORD *)a1 = v11;
    *(_BYTE *)(a1 + 8) = a3;
    v22 = 0;
    *(_QWORD *)(v11 + 64) = v4;
    v4 = 0;
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_BYTE *)(a1 + 8) = a3;
    v22 = 72LL * v5;
    if ( v5 )
    {
      v13 = v4;
      v24 = v4 + 72LL * v5;
      do
      {
        if ( *(_QWORD *)v13 != -8192 && *(_QWORD *)v13 != -4096 )
        {
          v14 = *(_QWORD **)(v13 + 8);
          v15 = &v14[3 * *(unsigned int *)(v13 + 16)];
          if ( v14 != v15 )
          {
            do
            {
              v16 = *(v15 - 1);
              v15 -= 3;
              if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
                sub_BD60C0(v15);
            }
            while ( v14 != v15 );
            v15 = *(_QWORD **)(v13 + 8);
          }
          if ( v15 != (_QWORD *)(v13 + 24) )
            _libc_free((unsigned __int64)v15);
        }
        v13 += 72;
      }
      while ( v24 != v13 );
    }
  }
  sub_C7D6A0(v4, v22, 8);
  return a1;
}
