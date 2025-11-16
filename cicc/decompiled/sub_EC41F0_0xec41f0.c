// Function: sub_EC41F0
// Address: 0xec41f0
//
__int64 __fastcall sub_EC41F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 result; // rax
  _DWORD *v9; // rdx
  __int64 v10; // rdi
  unsigned __int8 v11; // [rsp+Fh] [rbp-71h] BYREF
  _QWORD v12[4]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v13; // [rsp+30h] [rbp-50h]
  _QWORD v14[4]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v15; // [rsp+60h] [rbp-20h]

  v3 = *(_QWORD *)(a1 + 8);
  v11 = 2;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 40LL))(v3) + 8) != 2
    || (result = sub_EC3DC0(a1, (char *)&v11), !(_BYTE)result) )
  {
    v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    if ( v11 == 5 )
    {
      v14[0] = "cannot make section associative with .linkonce";
      v15 = 259;
    }
    else
    {
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 288) + 8LL);
      if ( (*(_BYTE *)(v5 + 149) & 0x10) == 0 )
      {
        sub_E93340(v5, v11);
        v9 = *(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
        result = 0;
        if ( *v9 != 9 )
        {
          v10 = *(_QWORD *)(a1 + 8);
          v14[0] = "unexpected token in directive";
          v15 = 259;
          return sub_ECE0E0(v10, v14, 0, 0);
        }
        return result;
      }
      v6 = *(_QWORD *)(v5 + 136);
      v7 = *(_QWORD *)(v5 + 128);
      v12[0] = "section '";
      v12[3] = v6;
      v13 = 1283;
      v12[2] = v7;
      v14[0] = v12;
      v14[2] = "' is already linkonce";
      v15 = 770;
    }
    return sub_ECDA70(*(_QWORD *)(a1 + 8), a2, v14, 0, 0);
  }
  return result;
}
