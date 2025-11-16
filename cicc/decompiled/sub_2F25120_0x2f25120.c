// Function: sub_2F25120
// Address: 0x2f25120
//
void __fastcall sub_2F25120(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r12
  char v5; // r13
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned __int64 v9; // rdi
  _QWORD v10[3]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v11; // [rsp+18h] [rbp-68h]
  __int64 v12; // [rsp+20h] [rbp-60h]
  unsigned int v13; // [rsp+28h] [rbp-58h]
  __int64 v14; // [rsp+30h] [rbp-50h]
  __int64 v15; // [rsp+38h] [rbp-48h]
  __int64 v16; // [rsp+40h] [rbp-40h]
  unsigned int v17; // [rsp+48h] [rbp-38h]

  v4 = *a3;
  v5 = *(_BYTE *)(*a3 + 128);
  sub_B2B9F0(*a3, unk_4F81788);
  v10[0] = a1;
  v10[1] = a2;
  v10[2] = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  sub_2F24820((__int64)v10, (__int64)a3);
  v6 = v17;
  if ( v17 )
  {
    v7 = v15;
    v8 = v15 + 48LL * v17;
    do
    {
      while ( 1 )
      {
        if ( (unsigned int)(*(_DWORD *)v7 + 0x7FFFFFFF) <= 0xFFFFFFFD )
        {
          v9 = *(_QWORD *)(v7 + 8);
          if ( v9 != v7 + 24 )
            break;
        }
        v7 += 48;
        if ( v8 == v7 )
          goto LABEL_7;
      }
      v7 += 48;
      j_j___libc_free_0(v9);
    }
    while ( v8 != v7 );
LABEL_7:
    v6 = v17;
  }
  sub_C7D6A0(v15, 48 * v6, 8);
  sub_C7D6A0(v11, 16LL * v13, 8);
  sub_B2B9F0(v4, v5);
}
