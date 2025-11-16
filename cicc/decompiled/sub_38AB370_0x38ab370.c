// Function: sub_38AB370
// Address: 0x38ab370
//
__int64 __fastcall sub_38AB370(__int64 **a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v9; // r12d
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-78h]
  __int64 v18; // [rsp+8h] [rbp-78h]
  unsigned __int64 v19; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v20; // [rsp+18h] [rbp-68h] BYREF
  __int64 v21; // [rsp+20h] [rbp-60h] BYREF
  __int64 v22; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v23[2]; // [rsp+30h] [rbp-50h] BYREF
  char v24; // [rsp+40h] [rbp-40h]
  char v25; // [rsp+41h] [rbp-3Fh]

  v7 = (unsigned __int64)a1[7];
  v20 = 0;
  v19 = v7;
  v8 = sub_38AB270(a1, &v21, a3, a4, a5, a6);
  if ( (_BYTE)v8 )
    return 1;
  if ( *(_BYTE *)(v21 + 16) != 18 )
  {
    v13 = *(_QWORD *)v21;
    if ( v13 != sub_1643320(*a1) )
    {
      v25 = 1;
      v24 = 3;
      v23[0] = "branch condition must have 'i1' type";
      return (unsigned int)sub_38814C0((__int64)(a1 + 1), v19, (__int64)v23);
    }
    if ( !(unsigned __int8)sub_388AF10((__int64)a1, 4, "expected ',' after branch condition")
      && !(unsigned __int8)sub_38AB2F0((__int64)a1, &v22, &v19, a3, a4, a5, a6)
      && !(unsigned __int8)sub_388AF10((__int64)a1, 4, "expected ',' after true destination") )
    {
      v9 = sub_38AB2F0((__int64)a1, v23, &v20, a3, a4, a5, a6);
      if ( !(_BYTE)v9 )
      {
        v14 = v23[0];
        v15 = v22;
        v18 = v21;
        v16 = sub_1648A60(56, 3u);
        v11 = v16;
        if ( v16 )
          sub_15F83E0((__int64)v16, v15, v14, v18, 0);
        goto LABEL_5;
      }
    }
    return 1;
  }
  v17 = v21;
  v9 = v8;
  v10 = sub_1648A60(56, 1u);
  v11 = v10;
  if ( v10 )
    sub_15F8320((__int64)v10, v17, 0);
LABEL_5:
  *a2 = v11;
  return v9;
}
