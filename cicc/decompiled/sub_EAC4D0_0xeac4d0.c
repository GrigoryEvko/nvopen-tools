// Function: sub_EAC4D0
// Address: 0xeac4d0
//
char __fastcall sub_EAC4D0(__int64 a1, __int64 *a2, __int64 a3)
{
  _QWORD **v6; // rdi
  __int64 (*v7)(void); // rax
  char v8; // al
  char result; // al
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // [rsp+8h] [rbp-C8h]
  __int128 v25; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+20h] [rbp-B0h]
  __int64 v27; // [rsp+28h] [rbp-A8h]
  __int64 v28; // [rsp+30h] [rbp-A0h]
  __int128 v29; // [rsp+40h] [rbp-90h]
  char v30; // [rsp+60h] [rbp-70h]
  char v31; // [rsp+61h] [rbp-6Fh]
  __int128 v32; // [rsp+70h] [rbp-60h] BYREF
  __int128 v33; // [rsp+80h] [rbp-50h]
  __int64 v34; // [rsp+90h] [rbp-40h]

  *a2 = 0;
  v6 = *(_QWORD ***)(a1 + 8);
  v7 = (__int64 (*)(void))(*v6)[3];
  if ( (char *)v7 == (char *)sub_EA2180 )
    v8 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, __int64, _QWORD))(*v6[1] + 240LL))(v6[1], a2, a3, 0);
  else
    v8 = v7();
  if ( v8 || (unsigned __int8)sub_EAC330(a1, 1u, a2, a3) )
    return 1;
  if ( (unsigned __int8)sub_ECE2A0(a1, 46) )
  {
    if ( **(_DWORD **)(a1 + 48) != 2 )
    {
      *(_QWORD *)&v32 = "unexpected symbol modifier following '@'";
      LOWORD(v34) = 259;
      return sub_ECE0E0(a1, &v32, 0, 0);
    }
    v11 = *(_QWORD *)(a1 + 240);
    v12 = sub_ECD7B0(a1);
    if ( *(_DWORD *)v12 == 2 )
    {
      v14 = *(_QWORD *)(v12 + 8);
      v13 = *(_QWORD *)(v12 + 16);
    }
    else
    {
      v13 = *(_QWORD *)(v12 + 16);
      v14 = *(_QWORD *)(v12 + 8);
      if ( v13 )
      {
        v15 = v13 - 1;
        if ( !v15 )
          v15 = 1;
        ++v14;
        v13 = v15 - 1;
      }
    }
    v24 = sub_106EF90(v11, v14, v13);
    if ( !BYTE4(v24) )
    {
      v31 = 1;
      *(_QWORD *)&v29 = "'";
      v30 = 3;
      v16 = sub_ECD7B0(a1);
      if ( *(_DWORD *)v16 == 2 )
      {
        v18 = *(_QWORD *)(v16 + 8);
        v17 = *(_QWORD *)(v16 + 16);
      }
      else
      {
        v17 = *(_QWORD *)(v16 + 16);
        v18 = *(_QWORD *)(v16 + 8);
        if ( v17 )
        {
          v19 = v17 - 1;
          if ( !v19 )
            v19 = 1;
          ++v18;
          v17 = v19 - 1;
        }
      }
      v26 = v18;
      *(_QWORD *)&v25 = "invalid variant '";
      LOWORD(v28) = 1283;
      v27 = v17;
      v33 = v29;
      *(_QWORD *)&v32 = &v25;
      LOBYTE(v34) = 2;
      BYTE1(v34) = v30;
      return sub_ECE0E0(a1, &v32, 0, 0);
    }
    v10 = sub_EA4970(a1, *a2, v24);
    if ( !v10 )
    {
      v31 = 1;
      *(_QWORD *)&v29 = "' (no symbols present)";
      v30 = 3;
      v20 = sub_ECD7B0(a1);
      if ( *(_DWORD *)v20 == 2 )
      {
        v22 = *(_QWORD *)(v20 + 8);
        v21 = *(_QWORD *)(v20 + 16);
      }
      else
      {
        v21 = *(_QWORD *)(v20 + 16);
        v22 = *(_QWORD *)(v20 + 8);
        if ( v21 )
        {
          v23 = v21 - 1;
          if ( !v23 )
            v23 = 1;
          ++v22;
          v21 = v23 - 1;
        }
      }
      v26 = v22;
      *(_QWORD *)&v25 = "invalid modifier '";
      LOWORD(v28) = 1283;
      v27 = v21;
      v33 = v29;
      *(_QWORD *)&v32 = &v25;
      LOBYTE(v34) = 2;
      BYTE1(v34) = v30;
      return sub_ECE0E0(a1, &v32, 0, 0);
    }
    *a2 = v10;
    sub_EABFE0(a1);
  }
  result = sub_E81180(*a2, &v32);
  if ( result )
  {
    *a2 = sub_E81A90(v32, *(_QWORD **)(a1 + 224), 0, 0);
    return 0;
  }
  return result;
}
