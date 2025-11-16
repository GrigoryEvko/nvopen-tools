// Function: sub_E48140
// Address: 0xe48140
//
__int64 __fastcall sub_E48140(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 *a5)
{
  char *v9; // rax
  __int64 v10; // rdi
  char v11; // al
  char *v12; // rax
  const char *v14; // rax
  __int64 v15; // r13
  _BYTE v16[32]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v18; // [rsp+30h] [rbp-70h]
  unsigned __int64 v19; // [rsp+38h] [rbp-68h]
  __int16 v20; // [rsp+40h] [rbp-60h]
  _QWORD v21[4]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v22; // [rsp+70h] [rbp-30h]

  v9 = (char *)sub_BA8B30(a2, a3, a4);
  if ( v9 )
  {
    v10 = (__int64)v9;
    v11 = *v9;
    if ( v11 == 1 )
    {
      v12 = (char *)sub_B325F0(v10);
      v10 = (__int64)v12;
      if ( !v12 )
      {
        v18 = a3;
        v17[0] = "Linking COMDATs named '";
        v21[0] = v17;
        v14 = "': COMDAT key involves incomputable alias size.";
        v20 = 1283;
        v19 = a4;
        goto LABEL_8;
      }
      v11 = *v12;
    }
    if ( v11 == 3 )
    {
      *a5 = v10;
      return 0;
    }
  }
  *a5 = 0;
  v20 = 1283;
  v18 = a3;
  v19 = a4;
  v17[0] = "Linking COMDATs named '";
  v21[0] = v17;
  v14 = "': GlobalVariable required for data dependent selection!";
LABEL_8:
  v21[2] = v14;
  v22 = 770;
  v15 = **(_QWORD **)(a1 + 8);
  sub_1061A30(v16, 0, v21);
  sub_B6EB20(v15, (__int64)v16);
  return 1;
}
