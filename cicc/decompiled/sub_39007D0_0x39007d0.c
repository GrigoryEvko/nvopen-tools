// Function: sub_39007D0
// Address: 0x39007d0
//
__int64 __fastcall sub_39007D0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdi
  unsigned int v7; // r12d
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  const char *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned __int64 v18; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v19[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v21; // [rsp+30h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v19[0] = 0;
  v19[1] = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v19);
  if ( (_BYTE)v3 )
  {
    HIBYTE(v21) = 1;
    v14 = "expected identifier in directive";
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 8);
    v18 = 0;
    v7 = v3;
    v8 = 0;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 40LL))(v6) + 8) == 12 )
    {
      v16 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
      v8 = sub_3909290(v16);
      v17 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64 *))(**(_QWORD **)(a1 + 8) + 200LL))(
              *(_QWORD *)(a1 + 8),
              &v18);
      if ( (_BYTE)v17 )
        return v17;
    }
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      v9 = *(_QWORD *)(a1 + 8);
      if ( v18 > 0xFFFFFFFF )
      {
        v20[0] = "invalid '.secrel32' directive offset, can't be less than zero or greater than std::numeric_limits<uint32_t>::max()";
        v21 = 259;
        return (unsigned int)sub_3909790(v9, v8, v20, 0, 0);
      }
      else
      {
        v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL))(v9);
        v20[0] = v19;
        v21 = 261;
        v11 = sub_38BF510(v10, (__int64)v20);
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
        v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
        (*(void (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v12 + 328LL))(v12, v11, v18);
      }
      return v7;
    }
    HIBYTE(v21) = 1;
    v14 = "unexpected token in directive";
  }
  v15 = *(_QWORD *)(a1 + 8);
  v20[0] = v14;
  LOBYTE(v21) = 3;
  return (unsigned int)sub_3909CF0(v15, v20, 0, 0, v4, v5);
}
