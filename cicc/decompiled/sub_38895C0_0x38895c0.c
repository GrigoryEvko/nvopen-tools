// Function: sub_38895C0
// Address: 0x38895c0
//
__int64 __fastcall sub_38895C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  bool v6; // zf
  unsigned __int64 v7; // rsi
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h]
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]
  _QWORD v17[2]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v18; // [rsp+40h] [rbp-20h]

  v5 = a1 + 8;
  v6 = *(_BYTE *)(a4 + 8) == 0;
  v13 = a2;
  v14 = a3;
  if ( v6 )
  {
    v10 = sub_3887100(v5);
    v11 = v14;
    v12 = v13;
    *(_DWORD *)(a1 + 64) = v10;
    return sub_3889300(a1, v12, v11, a4);
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 56);
    v16 = 1283;
    v15[0] = "field '";
    v15[1] = &v13;
    v17[0] = v15;
    v18 = 770;
    v17[1] = "' cannot be specified more than once";
    return sub_38814C0(v5, v7, (__int64)v17);
  }
}
