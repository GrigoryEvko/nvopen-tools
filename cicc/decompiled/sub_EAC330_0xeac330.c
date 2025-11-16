// Function: sub_EAC330
// Address: 0xeac330
//
__int64 __fastcall sub_EAC330(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4)
{
  unsigned int v6; // ebx
  _QWORD **v7; // rdi
  __int64 (__fastcall *v8)(__int64, __int64, __int64); // rax
  __int64 v9; // rdi
  int v10; // eax
  char v11; // dl
  unsigned int v12; // eax
  int *v13; // rax
  __int64 v14; // rdi
  int v15; // eax
  char v16; // dl
  __int64 i; // [rsp+10h] [rbp-50h]
  int v20; // [rsp+20h] [rbp-40h] BYREF
  int v21; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  for ( i = sub_ECD690(a1 + 40); ; *a3 = sub_E81A00(v20, *a3, v22[0], *(_QWORD **)(a1 + 224), i) )
  {
    v13 = *(int **)(a1 + 48);
    v14 = *(_QWORD *)(a1 + 240);
    v20 = 0;
    v15 = *v13;
    v16 = *(_BYTE *)(v14 + 400);
    if ( *(_BYTE *)(a1 + 868) )
    {
      v6 = sub_EA22E0(v15, &v20, v16);
      if ( a2 > v6 )
        return 0;
    }
    else
    {
      v6 = sub_EA3BB0(v14, v15, &v20, v16);
      if ( a2 > v6 )
        return 0;
    }
    sub_EABFE0(a1);
    v7 = *(_QWORD ***)(a1 + 8);
    v8 = (__int64 (__fastcall *)(__int64, __int64, __int64))(*v7)[3];
    if ( v8 == sub_EA2180 )
    {
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, __int64 *, __int64, _QWORD))(*v7[1] + 240LL))(v7[1], v22, a4, 0) )
        return 1;
    }
    else if ( (unsigned __int8)v8((__int64)v7, (__int64)v22, a4) )
    {
      return 1;
    }
    v9 = *(_QWORD *)(a1 + 240);
    v10 = **(_DWORD **)(a1 + 48);
    v11 = *(_BYTE *)(v9 + 400);
    v12 = *(_BYTE *)(a1 + 868) ? sub_EA22E0(v10, &v21, v11) : sub_EA3BB0(v9, v10, &v21, v11);
    if ( v6 < v12 && (unsigned __int8)sub_EAC330(a1, v6 + 1, v22, a4) )
      break;
  }
  return 1;
}
