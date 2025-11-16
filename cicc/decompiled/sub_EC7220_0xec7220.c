// Function: sub_EC7220
// Address: 0xec7220
//
__int64 __fastcall sub_EC7220(__int64 a1, _DWORD *a2, _BYTE *a3)
{
  __int64 v3; // r14
  __int64 v4; // r15
  bool v6; // zf
  const char **v7; // rdx
  char v8; // al
  char *v9; // rcx
  __int64 v10; // rdi
  __int64 v12; // rax
  bool v13; // cc
  _QWORD *v14; // rax
  const char **v15; // rdx
  const char *v16; // [rsp+0h] [rbp-90h] BYREF
  __int64 v17; // [rsp+8h] [rbp-88h]
  _BYTE *v18; // [rsp+10h] [rbp-80h]
  __int16 v19; // [rsp+20h] [rbp-70h]
  const char **v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+38h] [rbp-58h]
  char *v22; // [rsp+40h] [rbp-50h]
  char v23; // [rsp+50h] [rbp-40h]
  char v24; // [rsp+51h] [rbp-3Fh]

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 4 )
  {
    v6 = *a3 == 0;
    v16 = "invalid ";
    if ( v6 )
    {
      v19 = 259;
    }
    else
    {
      v18 = a3;
      v19 = 771;
    }
    v7 = &v16;
    v8 = 2;
    if ( HIBYTE(v19) == 1 )
    {
      v3 = v17;
      v7 = (const char **)v16;
      v8 = 3;
    }
    v20 = v7;
    v9 = " version number, integer expected";
    v21 = v3;
LABEL_7:
    v10 = *(_QWORD *)(a1 + 8);
    v22 = v9;
    v23 = v8;
    v24 = 3;
    return sub_ECE0E0(v10, &v20, 0, 0);
  }
  v12 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
  v13 = *(_DWORD *)(v12 + 32) <= 0x40u;
  v14 = *(_QWORD **)(v12 + 24);
  if ( !v13 )
    v14 = (_QWORD *)*v14;
  if ( (unsigned __int64)v14 > 0xFF )
  {
    v6 = *a3 == 0;
    v16 = "invalid ";
    if ( v6 )
    {
      v19 = 259;
    }
    else
    {
      v18 = a3;
      v19 = 771;
    }
    v15 = &v16;
    v8 = 2;
    if ( HIBYTE(v19) == 1 )
    {
      v4 = v17;
      v15 = (const char **)v16;
      v8 = 3;
    }
    v20 = v15;
    v9 = " version number";
    v21 = v4;
    goto LABEL_7;
  }
  *a2 = (_DWORD)v14;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  return 0;
}
