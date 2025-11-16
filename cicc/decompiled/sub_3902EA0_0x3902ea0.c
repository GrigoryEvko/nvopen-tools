// Function: sub_3902EA0
// Address: 0x3902ea0
//
__int64 __fastcall sub_3902EA0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // r13
  __int64 v12; // rax
  unsigned __int64 v14; // rdx
  const char *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  _QWORD v18[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v19[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v20; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v18[0] = 0;
  v18[1] = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v18);
  if ( (_BYTE)v3 )
  {
    v17 = *(_QWORD *)(a1 + 8);
    v19[0] = "expected identifier in directive";
    v20 = 259;
    return (unsigned int)sub_3909CF0(v17, v19, 0, 0, v4, v5);
  }
  else
  {
    v6 = v3;
    v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v19[0] = v18;
    v20 = 261;
    v8 = sub_38BF510(v7, (__int64)v19);
    v11 = (_QWORD *)v8;
    if ( (*(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL) != 0
      || (*(_BYTE *)(v8 + 9) & 0xC) == 8
      && (*(_BYTE *)(v8 + 8) |= 4u,
          v14 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v8 + 24)),
          *v11 = v14 | *v11 & 7LL,
          v14) )
    {
      HIBYTE(v20) = 1;
      v15 = ".alt_entry must preceed symbol definition";
    }
    else
    {
      v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)v12 + 256LL))(v12, v11, 16) )
      {
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
        return v6;
      }
      HIBYTE(v20) = 1;
      v15 = "unable to emit symbol attribute";
    }
    v16 = *(_QWORD *)(a1 + 8);
    v19[0] = v15;
    LOBYTE(v20) = 3;
    return (unsigned int)sub_3909CF0(v16, v19, 0, 0, v9, v10);
  }
}
