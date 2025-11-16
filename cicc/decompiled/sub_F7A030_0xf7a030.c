// Function: sub_F7A030
// Address: 0xf7a030
//
__int64 __fastcall sub_F7A030(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r8
  __int64 v5; // rbx
  __int64 v6; // r14
  char v7; // r15
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rdi
  char v11; // al
  unsigned __int64 v12; // rax
  __int64 v14; // [rsp+0h] [rbp-60h]
  unsigned __int64 v15; // [rsp+8h] [rbp-58h]

  v3 = a2 - a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
    v7 = *(_BYTE *)(v6 + 8);
    while ( 1 )
    {
      v8 = v5 >> 1;
      v9 = v4 + 8 * (v5 >> 1);
      v10 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
      v11 = *(_BYTE *)(v10 + 8);
      if ( v7 == 12 )
      {
        if ( v11 == 12 )
        {
          v14 = v4;
          v15 = sub_BCAE30(v10);
          v12 = sub_BCAE30(v6);
          v4 = v14;
          if ( v15 < v12 )
            goto LABEL_8;
        }
LABEL_4:
        v4 = v9 + 8;
        v5 = v5 - v8 - 1;
        if ( v5 <= 0 )
          return v4;
      }
      else
      {
        if ( v11 != 12 )
          goto LABEL_4;
LABEL_8:
        v5 >>= 1;
        if ( v8 <= 0 )
          return v4;
      }
    }
  }
  return v4;
}
