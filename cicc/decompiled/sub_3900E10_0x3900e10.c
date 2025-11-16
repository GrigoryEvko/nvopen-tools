// Function: sub_3900E10
// Address: 0x3900e10
//
__int64 __fastcall sub_3900E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // rdx
  _QWORD *v12; // rdx
  __int64 result; // rax
  _DWORD *v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdi
  int v18; // [rsp+Ch] [rbp-64h] BYREF
  _QWORD v19[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]
  _QWORD v22[2]; // [rsp+40h] [rbp-30h] BYREF
  __int16 v23; // [rsp+50h] [rbp-20h]

  v6 = *(_QWORD *)(a1 + 8);
  v18 = 2;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 40LL))(v6) + 8) != 2
    || (result = sub_3900970(a1, (__int64)&v18), !(_BYTE)result) )
  {
    v7 = 0;
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v9 = *(unsigned int *)(v8 + 120);
    if ( (_DWORD)v9 )
    {
      v10 = v18;
      v7 = *(_QWORD *)(*(_QWORD *)(v8 + 112) + 32 * v9 - 32);
      if ( v18 != 5 )
        goto LABEL_4;
    }
    else
    {
      v10 = v18;
      if ( v18 != 5 )
      {
LABEL_4:
        if ( (*(_BYTE *)(v7 + 169) & 0x10) == 0 )
        {
          sub_38D8560(v7, v10);
          v14 = *(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
          result = 0;
          if ( *v14 != 9 )
          {
            v17 = *(_QWORD *)(a1 + 8);
            v22[0] = "unexpected token in directive";
            v23 = 259;
            return sub_3909CF0(v17, v22, 0, 0, v15, v16);
          }
          return result;
        }
        v11 = *(_QWORD *)(v7 + 152);
        v19[1] = *(_QWORD *)(v7 + 160);
        v22[0] = "section '";
        v22[1] = v19;
        v23 = 1283;
        v20[0] = v22;
        v20[1] = "' is already linkonce";
        v19[0] = v11;
        v21 = 770;
        v12 = v20;
        return sub_3909790(*(_QWORD *)(a1 + 8), a4, v12, 0, 0);
      }
    }
    v12 = v22;
    v22[0] = "cannot make section associative with .linkonce";
    v23 = 259;
    return sub_3909790(*(_QWORD *)(a1 + 8), a4, v12, 0, 0);
  }
  return result;
}
