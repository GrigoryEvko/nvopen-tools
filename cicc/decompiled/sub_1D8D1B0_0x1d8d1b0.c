// Function: sub_1D8D1B0
// Address: 0x1d8d1b0
//
__int64 __fastcall sub_1D8D1B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  _BYTE *v4; // r13
  size_t v5; // r12
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 (*v8)(); // rax
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 *v11; // r13
  __int64 v12; // r8
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r12d
  _QWORD *v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v21; // [rsp+10h] [rbp-50h] BYREF
  __int64 v22; // [rsp+18h] [rbp-48h]
  _QWORD v23[8]; // [rsp+20h] [rbp-40h] BYREF

  v19 = sub_1560340((_QWORD *)(*(_QWORD *)a2 + 112LL), -1, "fentry-call", 0xBu);
  v2 = sub_155D8B0(&v19);
  if ( v2 )
  {
    v4 = (_BYTE *)v2;
    v20 = v3;
    v5 = v3;
    v21 = v23;
    v6 = v3;
    if ( v3 > 0xF )
    {
      v21 = (_QWORD *)sub_22409D0(&v21, &v20, 0);
      v18 = v21;
      v23[0] = v20;
    }
    else
    {
      if ( v3 == 1 )
      {
        LOBYTE(v23[0]) = *v4;
        v7 = v23;
LABEL_5:
        v22 = v6;
        *((_BYTE *)v7 + v6) = 0;
        goto LABEL_7;
      }
      if ( !v3 )
      {
        v7 = v23;
        goto LABEL_5;
      }
      v18 = v23;
    }
    memcpy(v18, v4, v5);
    v6 = v20;
    v7 = v21;
    goto LABEL_5;
  }
  v22 = 0;
  v21 = v23;
  LOBYTE(v23[0]) = 0;
LABEL_7:
  if ( (unsigned int)sub_2241AC0(&v21, "true") )
  {
    v16 = 0;
  }
  else
  {
    v8 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 40LL);
    if ( v8 == sub_1D00B00 )
      BUG();
    v9 = *(_QWORD *)(a2 + 328);
    v10 = *(_QWORD *)(v8() + 8);
    v20 = 0;
    v11 = *(__int64 **)(v9 + 32);
    v13 = sub_1E0B640(*(_QWORD *)(v9 + 56), v10 + 1280, &v20, 0, v12);
    sub_1DD5BA0(v9 + 16, v13);
    v14 = *v11;
    v15 = *(_QWORD *)v13;
    *(_QWORD *)(v13 + 8) = v11;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v13 = v14 | v15 & 7;
    *(_QWORD *)(v14 + 8) = v13;
    *v11 = *v11 & 7 | v13;
    if ( v20 )
      sub_161E7C0((__int64)&v20, v20);
    v16 = 1;
  }
  if ( v21 != v23 )
    j_j___libc_free_0(v21, v23[0] + 1LL);
  return v16;
}
