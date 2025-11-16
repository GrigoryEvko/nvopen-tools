// Function: sub_1DD5A70
// Address: 0x1dd5a70
//
__int64 __fastcall sub_1DD5A70(__int64 a1)
{
  __int64 v1; // rbp
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx
  _QWORD v9[2]; // [rsp-E8h] [rbp-E8h] BYREF
  _QWORD v10[2]; // [rsp-D8h] [rbp-D8h] BYREF
  __int16 v11; // [rsp-C8h] [rbp-C8h]
  __int64 v12; // [rsp-B8h] [rbp-B8h]
  _QWORD v13[2]; // [rsp-98h] [rbp-98h] BYREF
  __int16 v14; // [rsp-88h] [rbp-88h]
  _QWORD v15[2]; // [rsp-78h] [rbp-78h] BYREF
  __int16 v16; // [rsp-68h] [rbp-68h]
  __int64 v17; // [rsp-58h] [rbp-58h]
  _QWORD v18[2]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v19; // [rsp-28h] [rbp-28h]
  __int64 v20; // [rsp-8h] [rbp-8h]

  result = *(_QWORD *)(a1 + 192);
  if ( !result )
  {
    v20 = v1;
    v4 = *(_QWORD *)(a1 + 56);
    v5 = *(_QWORD *)(v4 + 24);
    LODWORD(v12) = *(_DWORD *)(v4 + 336);
    v6 = *(_QWORD *)(v5 + 16);
    v10[0] = v9;
    v10[1] = "BB";
    v7 = *(_QWORD *)(v6 + 104);
    v11 = 773;
    v8 = *(_QWORD *)(v6 + 96);
    v13[0] = v10;
    v9[0] = v8;
    LODWORD(v8) = *(_DWORD *)(a1 + 48);
    v13[1] = v12;
    v15[0] = v13;
    v15[1] = "_";
    LODWORD(v17) = v8;
    v18[0] = v15;
    v9[1] = v7;
    v19 = 2562;
    v14 = 2306;
    v16 = 770;
    v18[1] = v17;
    result = sub_38BF510(v5, v18);
    *(_QWORD *)(a1 + 192) = result;
  }
  return result;
}
