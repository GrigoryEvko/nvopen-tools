// Function: sub_C0A420
// Address: 0xc0a420
//
void __fastcall sub_C0A420(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v3; // r15
  _QWORD *v5; // rbx
  __int64 v6; // rdi
  _BYTE *v7; // rax
  _QWORD **v8; // rax
  __int64 v9; // r14
  __int64 *v10; // rsi
  _QWORD v11[8]; // [rsp-198h] [rbp-198h] BYREF
  _BYTE *v12; // [rsp-158h] [rbp-158h] BYREF
  size_t v13; // [rsp-150h] [rbp-150h]
  __int64 v14; // [rsp-148h] [rbp-148h]
  _BYTE v15[320]; // [rsp-140h] [rbp-140h] BYREF

  if ( a3 )
  {
    v3 = a2;
    v5 = &a2[4 * a3];
    v12 = v15;
    v11[5] = 0x100000000LL;
    v11[0] = &unk_49DD288;
    v13 = 0;
    v14 = 256;
    v11[1] = 2;
    memset(&v11[2], 0, 24);
    v11[6] = &v12;
    sub_CB5980(v11, 0, 0, 0);
    if ( a2 != v5 )
    {
      do
      {
        while ( 1 )
        {
          v6 = sub_CB6200(v11, *v3, v3[1]);
          v7 = *(_BYTE **)(v6 + 32);
          if ( *(_BYTE **)(v6 + 24) == v7 )
            break;
          v3 += 4;
          *v7 = 44;
          ++*(_QWORD *)(v6 + 32);
          if ( v5 == v3 )
            goto LABEL_7;
        }
        v3 += 4;
        sub_CB6200(v6, ",", 1);
      }
      while ( v5 != v3 );
    }
LABEL_7:
    --v13;
    v8 = (_QWORD **)sub_B43CA0(a1);
    v9 = sub_A78730(*v8, "vector-function-abi-variant", 0x1Bu, v12, v13);
    v10 = (__int64 *)sub_BD5C60(a1);
    *(_QWORD *)(a1 + 72) = sub_A7B440((__int64 *)(a1 + 72), v10, -1, v9);
    v11[0] = &unk_49DD388;
    sub_CB5840(v11);
    if ( v12 != v15 )
      _libc_free(v12, v10);
  }
}
