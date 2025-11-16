// Function: sub_38EB510
// Address: 0x38eb510
//
__int64 __fastcall sub_38EB510(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4)
{
  unsigned int v6; // ebx
  _QWORD **v7; // rdi
  __int64 (__fastcall *v8)(__int64); // rax
  int v9; // edi
  char v10; // dl
  unsigned int v11; // eax
  int *v12; // rax
  int v13; // edi
  char v14; // dl
  __int64 i; // [rsp+10h] [rbp-50h]
  int v18; // [rsp+20h] [rbp-40h] BYREF
  int v19; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v20[7]; // [rsp+28h] [rbp-38h] BYREF

  for ( i = sub_3909290(a1 + 144); ; *a3 = sub_38CB1F0(v18, *a3, v20[0], *(_QWORD *)(a1 + 320), i) )
  {
    v12 = *(int **)(a1 + 152);
    v18 = 0;
    v13 = *v12;
    v14 = *(_BYTE *)(*(_QWORD *)(a1 + 336) + 400LL);
    if ( *(_BYTE *)(a1 + 844) )
    {
      v6 = sub_38E2AE0(v13, &v18, v14);
      if ( a2 > v6 )
        return 0;
    }
    else
    {
      v6 = sub_38E2C30(v13, &v18, v14);
      if ( a2 > v6 )
        return 0;
    }
    sub_38EB180(a1);
    v7 = *(_QWORD ***)(a1 + 8);
    v8 = (__int64 (__fastcall *)(__int64))(*v7)[3];
    if ( v8 == sub_38E29C0 )
    {
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, __int64 *, __int64))(*v7[1] + 184LL))(v7[1], v20, a4) )
        return 1;
    }
    else if ( ((unsigned __int8 (__fastcall *)(_QWORD **, __int64 *, __int64))v8)(v7, v20, a4) )
    {
      return 1;
    }
    v9 = **(_DWORD **)(a1 + 152);
    v10 = *(_BYTE *)(*(_QWORD *)(a1 + 336) + 400LL);
    v11 = *(_BYTE *)(a1 + 844) ? sub_38E2AE0(v9, &v19, v10) : sub_38E2C30(v9, &v19, v10);
    if ( v6 < v11 && (unsigned __int8)sub_38EB510(a1, v6 + 1, v20, a4) )
      break;
  }
  return 1;
}
