// Function: sub_145DC80
// Address: 0x145dc80
//
__int64 __fastcall sub_145DC80(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // [rsp+18h] [rbp-D8h]
  __int64 v12; // [rsp+18h] [rbp-D8h]
  __int64 v13; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v14[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v15[176]; // [rsp+40h] [rbp-B0h] BYREF

  v14[1] = 0x2000000000LL;
  v14[0] = (unsigned __int64)v15;
  sub_16BD3E0(v14, 10);
  sub_16BD4C0(v14, a2);
  v13 = 0;
  v2 = sub_16BDDE0(a1 + 816, v14, &v13);
  if ( !v2 )
  {
    v4 = sub_16BD760(v14, a1 + 864);
    v6 = v5;
    v11 = v4;
    v7 = sub_145CBF0((__int64 *)(a1 + 864), 80, 16);
    v8 = a1 + 816;
    v9 = v7;
    v10 = *(_QWORD *)(a1 + 1032);
    *(_QWORD *)(v9 + 40) = v11;
    *(_QWORD *)(v9 + 32) = 0;
    *(_QWORD *)(v9 + 48) = v6;
    *(_DWORD *)(v9 + 56) = 10;
    *(_QWORD *)(v9 + 8) = 2;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)(v9 + 24) = a2;
    if ( a2 != -8 && a2 != 0 && a2 != -16 )
    {
      v12 = v10;
      sub_164C220(v9 + 8);
      v8 = a1 + 816;
      v10 = v12;
    }
    *(_QWORD *)(v9 + 64) = a1;
    v2 = v9 + 32;
    *(_QWORD *)(v9 + 72) = v10;
    *(_QWORD *)(a1 + 1032) = v9;
    *(_QWORD *)v9 = &unk_49EC5A0;
    sub_16BDA20(v8, v9 + 32, v13);
  }
  if ( (_BYTE *)v14[0] != v15 )
    _libc_free(v14[0]);
  return v2;
}
