// Function: sub_38EBA90
// Address: 0x38eba90
//
__int64 __fastcall sub_38EBA90(__int64 a1, char a2, unsigned int a3)
{
  __int64 v6; // rax
  bool v7; // zf
  unsigned int v8; // r15d
  unsigned __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  char v17; // al
  __int64 v18; // rdx
  int v19; // eax
  char v20; // [rsp+6h] [rbp-7Ah]
  unsigned __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-68h] BYREF
  __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  __int64 v25; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v26[2]; // [rsp+30h] [rbp-50h] BYREF
  char v27; // [rsp+40h] [rbp-40h]
  char v28; // [rsp+41h] [rbp-3Fh]

  v6 = sub_3909290(a1 + 144);
  v7 = *(_BYTE *)(a1 + 845) == 0;
  v23 = 0;
  v21 = v6;
  v24 = 0;
  v25 = 0;
  if ( v7 && (unsigned __int8)sub_38E36C0(a1) )
  {
    v28 = 1;
    v26[0] = " in directive";
    v27 = 3;
    return (unsigned int)sub_39094A0(a1, v26);
  }
  if ( a3 == 1 && a2 && *(_DWORD *)sub_3909460(a1) == 9 )
  {
    v28 = 1;
    v26[0] = "p2align directive with no operand(s) is ignored";
    v27 = 3;
    sub_38E4170((_QWORD *)a1, v21, (__int64)v26, 0, 0);
    v28 = 1;
    v26[0] = "unexpected token";
    v27 = 3;
    return (unsigned int)sub_3909E20(a1, 9, v26);
  }
  if ( (unsigned __int8)sub_38EB9C0(a1, &v22) )
    goto LABEL_7;
  v20 = sub_3909EB0(a1, 25);
  if ( v20 )
  {
    if ( *(_DWORD *)sub_3909460(a1) == 25 )
    {
      v20 = 0;
    }
    else if ( (unsigned __int8)sub_38EB9C0(a1, &v24) )
    {
LABEL_7:
      v28 = 1;
      v26[0] = " in directive";
      v27 = 3;
      return (unsigned int)sub_39094A0(a1, v26);
    }
    if ( (unsigned __int8)sub_3909EB0(a1, 25)
      && ((unsigned __int8)sub_3909470(a1, &v23) || (unsigned __int8)sub_38EB9C0(a1, &v25)) )
    {
      goto LABEL_7;
    }
  }
  v10 = 9;
  v28 = 1;
  v27 = 3;
  v26[0] = "unexpected token";
  v8 = sub_3909E20(a1, 9, v26);
  if ( (_BYTE)v8 )
    goto LABEL_7;
  v11 = v22;
  if ( a2 )
  {
    if ( v22 > 31 )
    {
      v10 = v21;
      v28 = 1;
      v26[0] = "invalid alignment value";
      v27 = 3;
      v8 = sub_3909790(a1, v21, v26, 0, 0);
      v12 = 0x80000000LL;
    }
    else
    {
      v12 = 1LL << v22;
    }
    v22 = v12;
  }
  else if ( v22 )
  {
    if ( (v22 & (v22 - 1)) != 0 )
    {
      v10 = v21;
      v28 = 1;
      v26[0] = "alignment must be a power of 2";
      v27 = 3;
      v8 = sub_3909790(a1, v21, v26, 0, 0);
    }
  }
  else
  {
    v22 = 1;
  }
  if ( v23 )
  {
    v13 = v25;
    if ( v25 <= 0 )
    {
      v10 = v23;
      v28 = 1;
      v26[0] = "alignment directive can never be satisfied in this many bytes, ignoring maximum bytes expression";
      v27 = 3;
      v19 = sub_3909790(a1, v23, v26, 0, 0);
      v25 = 0;
      v8 |= v19;
      v13 = 0;
    }
    if ( v22 <= v13 )
    {
      v10 = v23;
      v28 = 1;
      v26[0] = "maximum bytes expression exceeds alignment and has no effect";
      v27 = 3;
      sub_38E4170((_QWORD *)a1, v23, (__int64)v26, 0, 0);
      v25 = 0;
    }
  }
  v14 = *(_QWORD *)(a1 + 328);
  v15 = *(unsigned int *)(v14 + 120);
  if ( !(_DWORD)v15 )
    BUG();
  v16 = *(_QWORD *)(*(_QWORD *)(v14 + 112) + 32 * v15 - 32);
  v17 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64))(*(_QWORD *)v16 + 8LL))(
          v16,
          v10,
          v14,
          v11);
  if ( !v20 || (v18 = v24, *(_DWORD *)(*(_QWORD *)(a1 + 280) + 284LL) == v24) )
  {
    if ( a3 == 1 && v17 )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 328) + 520LL))(
        *(_QWORD *)(a1 + 328),
        (unsigned int)v22,
        (unsigned int)v25);
      return v8;
    }
    v18 = v24;
  }
  (*(void (__fastcall **)(_QWORD, _QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 328) + 512LL))(
    *(_QWORD *)(a1 + 328),
    (unsigned int)v22,
    v18,
    a3,
    (unsigned int)v25);
  return v8;
}
