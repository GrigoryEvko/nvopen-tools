// Function: sub_3205D70
// Address: 0x3205d70
//
__int64 __fastcall sub_3205D70(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // r14
  unsigned int v7; // eax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r12
  void (*v14)(); // rax
  __int64 v15; // [rsp+8h] [rbp-78h]
  char *v16; // [rsp+20h] [rbp-60h] BYREF
  char v17; // [rsp+40h] [rbp-40h]
  char v18; // [rsp+41h] [rbp-3Fh]

  result = a2[1];
  v5 = *a2;
  v15 = result;
  if ( *a2 != result )
  {
    v6 = *a2;
    do
    {
      v10 = *(_QWORD *)(v6 + 32);
      v11 = sub_31F8790(a1, 4360, a3, a4, v5);
      v12 = *(_QWORD *)(a1 + 528);
      v13 = v11;
      v14 = *(void (**)())(*(_QWORD *)v12 + 120LL);
      v18 = 1;
      v16 = "Type";
      v17 = 3;
      if ( v14 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, char **, __int64))v14)(v12, &v16, 1);
        v12 = *(_QWORD *)(a1 + 528);
      }
      if ( v10 )
      {
        v7 = sub_3205010(a1, v10);
      }
      else
      {
        LODWORD(v16) = 3;
        v7 = 3;
      }
      v6 += 40;
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v12 + 536LL))(v12, v7, 4);
      sub_31F4F00(*(__int64 **)(a1 + 528), *(const void **)(v6 - 40), *(_QWORD *)(v6 - 32), 3840, v8, v9);
      result = sub_31F8930(a1, v13);
    }
    while ( v15 != v6 );
  }
  return result;
}
