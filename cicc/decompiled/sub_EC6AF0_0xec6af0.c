// Function: sub_EC6AF0
// Address: 0xec6af0
//
__int64 __fastcall sub_EC6AF0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  _QWORD *v10; // r12
  __int64 result; // rax
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  char v18; // cl
  __int64 v19; // rdi
  __int64 v20; // [rsp+8h] [rbp-108h]
  __int64 v21; // [rsp+10h] [rbp-100h]
  char v23; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v24; // [rsp+30h] [rbp-E0h]
  __int64 v25; // [rsp+38h] [rbp-D8h]
  __int16 v26; // [rsp+40h] [rbp-D0h]
  _QWORD *v27; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+58h] [rbp-B8h]
  char *v29; // [rsp+60h] [rbp-B0h]
  __int64 v30; // [rsp+68h] [rbp-A8h]
  __int16 v31; // [rsp+70h] [rbp-A0h]
  _QWORD v32[4]; // [rsp+80h] [rbp-90h] BYREF
  char v33; // [rsp+A0h] [rbp-70h]
  char v34; // [rsp+A1h] [rbp-6Fh]
  _QWORD v35[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v36; // [rsp+D0h] [rbp-40h]

  v10 = a2;
  result = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  if ( a7 != *(_DWORD *)(result + 68) )
  {
    v16 = sub_CC7380((__int64 *)(result + 24));
    v27 = a2;
    if ( a5 )
    {
      v10 = &v27;
      v23 = 32;
      v24 = a4;
      v26 = 1288;
      v29 = &v23;
      v25 = a5;
      v30 = v20;
      v18 = 2;
      v28 = a3;
      v31 = 517;
    }
    else
    {
      v18 = 5;
      v26 = 257;
      v28 = a3;
      v31 = 261;
      v21 = a3;
    }
    v19 = *(_QWORD *)(a1 + 8);
    v33 = v18;
    v35[2] = v16;
    v32[1] = v21;
    v32[2] = " used while targeting ";
    v35[0] = v32;
    v35[3] = v17;
    v32[0] = v10;
    v34 = 3;
    v35[1] = 0;
    v36 = 1282;
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v19 + 168LL))(
               v19,
               a6,
               v35,
               0,
               0);
  }
  if ( *(_QWORD *)(a1 + 24) )
  {
    v13 = *(_QWORD *)(a1 + 8);
    v35[0] = "overriding previous version directive";
    v36 = 259;
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v13 + 168LL))(v13, a6, v35, 0, 0);
    v14 = *(_QWORD *)(a1 + 8);
    v36 = 259;
    v15 = *(_QWORD *)(a1 + 24);
    v35[0] = "previous definition is here";
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v14 + 160LL))(
               v14,
               v15,
               v35,
               0,
               0);
  }
  *(_QWORD *)(a1 + 24) = a6;
  return result;
}
