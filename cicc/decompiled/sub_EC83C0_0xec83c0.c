// Function: sub_EC83C0
// Address: 0xec83c0
//
__int64 __fastcall sub_EC83C0(__int64 a1, _DWORD *a2, _DWORD *a3, const char *a4)
{
  __int64 v4; // r14
  bool v7; // zf
  const char **v8; // rdx
  char v9; // al
  const char *v10; // rbx
  __int64 v12; // rax
  bool v13; // cc
  _QWORD *v14; // rax
  const char **v15; // rdx
  const char **v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rax
  const char **v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-A8h]
  __int64 v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+18h] [rbp-98h]
  const char *v23; // [rsp+20h] [rbp-90h] BYREF
  __int64 v24; // [rsp+28h] [rbp-88h]
  const char *v25; // [rsp+30h] [rbp-80h]
  __int16 v26; // [rsp+40h] [rbp-70h]
  const char *v27; // [rsp+50h] [rbp-60h] BYREF
  __int64 v28; // [rsp+58h] [rbp-58h]
  const char *v29; // [rsp+60h] [rbp-50h]
  __int16 v30; // [rsp+70h] [rbp-40h]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 4 )
  {
    v7 = *a4 == 0;
    v23 = "invalid ";
    if ( v7 )
    {
      v26 = 259;
    }
    else
    {
      v25 = a4;
      v26 = 771;
    }
    v8 = &v23;
    v9 = 2;
    if ( HIBYTE(v26) == 1 )
    {
      v4 = v24;
      v8 = (const char **)v23;
      v9 = 3;
    }
    v27 = (const char *)v8;
    v10 = " major version number, integer expected";
    v28 = v4;
LABEL_7:
    v29 = v10;
LABEL_8:
    LOBYTE(v30) = v9;
    HIBYTE(v30) = 3;
    return sub_ECE0E0(*(_QWORD *)(a1 + 8), &v27, 0, 0);
  }
  v12 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
  v13 = *(_DWORD *)(v12 + 32) <= 0x40u;
  v14 = *(_QWORD **)(v12 + 24);
  if ( !v13 )
    v14 = (_QWORD *)*v14;
  if ( (unsigned __int64)v14 - 1 > 0xFFFE )
  {
    v7 = *a4 == 0;
    v23 = "invalid ";
    if ( v7 )
    {
      v26 = 259;
    }
    else
    {
      v25 = a4;
      v26 = 771;
    }
    v15 = &v23;
    v9 = 2;
    if ( HIBYTE(v26) == 1 )
    {
      v15 = (const char **)v23;
      v22 = v24;
      v9 = 3;
    }
    v27 = (const char *)v15;
    v29 = " major version number";
    v28 = v22;
    goto LABEL_8;
  }
  *a2 = (_DWORD)v14;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
  {
    if ( *a4 )
    {
      v27 = a4;
      v29 = " minor version number required, comma expected";
      v30 = 771;
    }
    else
    {
      v27 = " minor version number required, comma expected";
      v30 = 259;
    }
    return sub_ECE0E0(*(_QWORD *)(a1 + 8), &v27, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 4 )
  {
    v7 = *a4 == 0;
    v23 = "invalid ";
    if ( v7 )
    {
      v26 = 259;
    }
    else
    {
      v25 = a4;
      v26 = 771;
    }
    v16 = &v23;
    v9 = 2;
    if ( HIBYTE(v26) == 1 )
    {
      v16 = (const char **)v23;
      v21 = v24;
      v9 = 3;
    }
    v27 = (const char *)v16;
    v28 = v21;
    v10 = " minor version number, integer expected";
    goto LABEL_7;
  }
  v17 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
  v13 = *(_DWORD *)(v17 + 32) <= 0x40u;
  v18 = *(_QWORD **)(v17 + 24);
  if ( !v13 )
    v18 = (_QWORD *)*v18;
  if ( (unsigned __int64)v18 > 0xFF )
  {
    v7 = *a4 == 0;
    v23 = "invalid ";
    if ( v7 )
    {
      v26 = 259;
    }
    else
    {
      v25 = a4;
      v26 = 771;
    }
    v19 = &v23;
    v9 = 2;
    if ( HIBYTE(v26) == 1 )
    {
      v19 = (const char **)v23;
      v20 = v24;
      v9 = 3;
    }
    v27 = (const char *)v19;
    v28 = v20;
    v10 = " minor version number";
    goto LABEL_7;
  }
  *a3 = (_DWORD)v18;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  return 0;
}
