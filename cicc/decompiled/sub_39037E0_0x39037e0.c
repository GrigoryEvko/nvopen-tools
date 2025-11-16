// Function: sub_39037E0
// Address: 0x39037e0
//
__int64 __fastcall sub_39037E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, unsigned int a7)
{
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 result; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rdx
  char v16; // dl
  _QWORD *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v20; // [rsp+8h] [rbp-E8h]
  _QWORD v21[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v22[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v23; // [rsp+30h] [rbp-C0h]
  _QWORD v24[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v25; // [rsp+60h] [rbp-90h]
  _QWORD v26[2]; // [rsp+70h] [rbp-80h] BYREF
  __int16 v27; // [rsp+80h] [rbp-70h]
  _QWORD v28[2]; // [rsp+90h] [rbp-60h] BYREF
  char v29; // [rsp+A0h] [rbp-50h]
  char v30; // [rsp+A1h] [rbp-4Fh]
  _QWORD v31[2]; // [rsp+B0h] [rbp-40h] BYREF
  __int16 v32; // [rsp+C0h] [rbp-30h]

  v9 = *(_QWORD *)(a1 + 8);
  v21[0] = a2;
  v21[1] = a3;
  v19 = a4;
  v20 = a5;
  v10 = *(_QWORD *)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL))(v9) + 32);
  result = a7;
  if ( *(_DWORD *)(v10 + 740) != a7 )
  {
    v22[0] = sub_16E2000((__int64 *)(v10 + 696));
    v22[1] = v15;
    if ( v20 )
    {
      LOBYTE(v23) = 32;
      v25 = 1288;
      v16 = 2;
      v24[0] = v23;
      v24[1] = &v19;
      v26[0] = v21;
      v26[1] = v24;
      v17 = v26;
      v27 = 517;
    }
    else
    {
      v16 = 5;
      v17 = v21;
      v25 = 257;
      v26[0] = v21;
      v27 = 261;
    }
    v28[0] = v17;
    v28[1] = " used while targeting ";
    v18 = *(_QWORD *)(a1 + 8);
    v31[0] = v28;
    v31[1] = v22;
    v29 = v16;
    v30 = 3;
    v32 = 1282;
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v18 + 120LL))(
               v18,
               a6,
               v31,
               0,
               0);
  }
  if ( *(_QWORD *)(a1 + 24) )
  {
    v12 = *(_QWORD *)(a1 + 8);
    v31[0] = "overriding previous version directive";
    v32 = 259;
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v12 + 120LL))(v12, a6, v31, 0, 0);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = *(_QWORD *)(a1 + 24);
    v31[0] = "previous definition is here";
    v32 = 259;
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v13 + 112LL))(
               v13,
               v14,
               v31,
               0,
               0);
  }
  *(_QWORD *)(a1 + 24) = a6;
  return result;
}
