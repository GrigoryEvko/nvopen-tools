// Function: sub_35EE840
// Address: 0x35ee840
//
void __fastcall sub_35EE840(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _BYTE v12[8]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v13; // [rsp+8h] [rbp-58h]
  _QWORD v14[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 16) + 16LL * a3;
  if ( *(_BYTE *)v6 == 1 )
  {
    (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)a1 + 40LL))(a1, a4, *(unsigned int *)(v6 + 8));
  }
  else if ( *(_BYTE *)v6 == 2 )
  {
    sub_E82C90((__int64)v12, a1, a4, 0);
    v7 = *(_QWORD *)(v6 + 8);
    if ( *(_BYTE *)(a1 + 51) )
      sub_E82920(v14, a1, v7);
    else
      sub_E828F0(v14, a1, v7);
    sub_CB6620(v13, (__int64)v14, v8, v9, v10, v11);
    sub_E82CC0((__int64)v12);
  }
  else
  {
    sub_E7FAD0(*(unsigned int **)(v6 + 8), (__int64)a4, *(_QWORD *)(a1 + 16), 0, (__int64)a4, a6);
  }
}
