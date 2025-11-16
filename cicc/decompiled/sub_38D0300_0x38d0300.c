// Function: sub_38D0300
// Address: 0x38d0300
//
__int64 __fastcall sub_38D0300(_QWORD *a1, __int64 a2, char a3, _QWORD *a4)
{
  __int64 v8; // rdi
  unsigned int v9; // eax
  unsigned int v10; // r12d
  __int64 v11; // rbx
  __int64 *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD v16[2]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v18; // [rsp+20h] [rbp-80h]
  _QWORD v19[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v20; // [rsp+40h] [rbp-60h]
  __int64 v21; // [rsp+50h] [rbp-50h] BYREF
  __int64 v22; // [rsp+58h] [rbp-48h]
  __int64 v23; // [rsp+60h] [rbp-40h]
  int v24; // [rsp+68h] [rbp-38h]

  *(_BYTE *)(a2 + 8) |= 4u;
  v8 = *(_QWORD *)(a2 + 24);
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v9 = sub_38CF2F0(v8, (__int64)&v21, a1);
  if ( !(_BYTE)v9 )
  {
    if ( (*(_BYTE *)a2 & 4) != 0 )
    {
      v13 = *(__int64 **)(a2 - 8);
      v14 = *v13;
      v15 = v13 + 2;
    }
    else
    {
      v14 = 0;
      v15 = 0;
    }
    v16[0] = v15;
    v17[0] = "unable to evaluate offset for variable '";
    v17[1] = v16;
    v19[0] = v17;
    v16[1] = v14;
    v18 = 1283;
    v19[1] = "'";
    v20 = 770;
    sub_16BCFB0((__int64)v19, 1u);
  }
  v10 = v9;
  v11 = v23;
  if ( v21 )
  {
    if ( !(unsigned __int8)sub_38D01D0((__int64)a1, *(_QWORD *)(v21 + 24), a3, v19) )
      return 0;
    v11 += v19[0];
  }
  if ( v22 )
  {
    if ( (unsigned __int8)sub_38D01D0((__int64)a1, *(_QWORD *)(v22 + 24), a3, v19) )
    {
      v11 -= v19[0];
      goto LABEL_8;
    }
    return 0;
  }
LABEL_8:
  *a4 = v11;
  return v10;
}
