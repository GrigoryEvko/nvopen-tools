// Function: sub_35BC650
// Address: 0x35bc650
//
__int64 __fastcall sub_35BC650(__int64 a1, __int64 a2, __int64 a3, int *a4, __int64 (__fastcall *a5)(__int64, __int64))
{
  __int64 i; // r14
  __int64 v6; // r13
  __int64 v7; // rbx
  char v8; // al
  bool v9; // zf
  __int64 v10; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v17; // [rsp+18h] [rbp-78h]
  __int64 (__fastcall *v19)(__int64, __int64); // [rsp+38h] [rbp-58h] BYREF
  int v20; // [rsp+40h] [rbp-50h] BYREF
  __int64 v21; // [rsp+48h] [rbp-48h]
  __int64 v22; // [rsp+50h] [rbp-40h]

  v17 = (a3 - 1) / 2;
  if ( a2 >= v17 )
  {
    v6 = a2;
  }
  else
  {
    for ( i = a2; ; i = v6 )
    {
      v6 = 2 * (i + 1);
      v7 = a1 + 48 * (i + 1);
      v8 = ((__int64 (__fastcall *)(__int64))a5)(v7);
      v9 = v8 == 0;
      if ( v8 )
        v7 = a1 + 24 * (v6 - 1);
      v10 = a1 + 24 * i;
      if ( !v9 )
        --v6;
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(v7 + 16);
      *(_QWORD *)(v10 + 8) = *(_QWORD *)(v7 + 8);
      *(_DWORD *)v10 = *(_DWORD *)v7;
      if ( v6 >= v17 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v6 )
  {
    v12 = a1 + 24 * (2 * v6 + 1);
    v13 = 3 * v6;
    v6 = 2 * v6 + 1;
    v14 = a1 + 8 * v13;
    *(_QWORD *)(v14 + 16) = *(_QWORD *)(v12 + 16);
    *(_QWORD *)(v14 + 8) = *(_QWORD *)(v12 + 8);
    *(_DWORD *)v14 = *(_DWORD *)v12;
  }
  v19 = a5;
  v20 = *a4;
  v21 = *((_QWORD *)a4 + 1);
  v22 = *((_QWORD *)a4 + 2);
  return sub_35BB8E0(a1, v6, a2, (__int64)&v20, &v19);
}
