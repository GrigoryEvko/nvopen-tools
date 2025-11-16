// Function: sub_2E30F60
// Address: 0x2e30f60
//
__int64 __fastcall sub_2E30F60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  __int64 result; // rax
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rdi
  const char *v12; // [rsp-A8h] [rbp-A8h] BYREF
  int v13; // [rsp-98h] [rbp-98h]
  __int16 v14; // [rsp-88h] [rbp-88h]
  _QWORD v15[4]; // [rsp-78h] [rbp-78h] BYREF
  __int16 v16; // [rsp-58h] [rbp-58h]
  _QWORD *v17; // [rsp-48h] [rbp-48h] BYREF
  int v18; // [rsp-38h] [rbp-38h]
  __int16 v19; // [rsp-28h] [rbp-28h]
  __int64 v20; // [rsp-8h] [rbp-8h]

  result = *(_QWORD *)(a1 + 280);
  if ( !result )
  {
    v20 = v6;
    v9 = *(_QWORD *)(a1 + 32);
    v10 = *(_DWORD *)(a1 + 24);
    v12 = "BB_END";
    v11 = *(_QWORD *)(v9 + 24);
    LODWORD(v9) = *(_DWORD *)(v9 + 336);
    v18 = v10;
    v19 = 2562;
    v13 = v9;
    v14 = 2307;
    v15[0] = &v12;
    v15[2] = "_";
    v16 = 770;
    v17 = v15;
    result = sub_E6C900(v11, (__int64 *)&v17, 0, 770, a5, a6);
    *(_QWORD *)(a1 + 280) = result;
  }
  return result;
}
