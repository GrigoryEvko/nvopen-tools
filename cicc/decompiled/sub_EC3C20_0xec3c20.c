// Function: sub_EC3C20
// Address: 0xec3c20
//
__int64 __fastcall sub_EC3C20(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rdi
  unsigned int v5; // r12d
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rax
  const char *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned __int64 v16; // [rsp+8h] [rbp-68h] BYREF
  const char *v17; // [rsp+10h] [rbp-60h] BYREF
  const char *v18; // [rsp+18h] [rbp-58h]
  const char *v19[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v20; // [rsp+40h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v17 = 0;
  v18 = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v17);
  if ( (_BYTE)v3 )
  {
    HIBYTE(v20) = 1;
    v12 = "expected identifier in directive";
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 8);
    v16 = 0;
    v5 = v3;
    v6 = 0;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 40LL))(v4) + 8) == 12 )
    {
      v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
      v6 = sub_ECD690(v14);
      v15 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64 *))(**(_QWORD **)(a1 + 8) + 256LL))(
              *(_QWORD *)(a1 + 8),
              &v16);
      if ( (_BYTE)v15 )
        return v15;
    }
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      v7 = *(_QWORD *)(a1 + 8);
      if ( v16 > 0xFFFFFFFF )
      {
        v19[0] = "invalid '.secrel32' directive offset, can't be less than zero or greater than std::numeric_limits<uint32_t>::max()";
        v20 = 259;
        return (unsigned int)sub_ECDA70(v7, v6, v19, 0, 0);
      }
      else
      {
        v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 48LL))(v7);
        v20 = 261;
        v19[0] = v17;
        v19[1] = v18;
        v9 = sub_E6C460(v8, v19);
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
        v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
        (*(void (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v10 + 368LL))(v10, v9, v16);
      }
      return v5;
    }
    HIBYTE(v20) = 1;
    v12 = "unexpected token in directive";
  }
  v13 = *(_QWORD *)(a1 + 8);
  v19[0] = v12;
  LOBYTE(v20) = 3;
  return (unsigned int)sub_ECE0E0(v13, v19, 0, 0);
}
