// Function: sub_38EC670
// Address: 0x38ec670
//
__int64 __fastcall sub_38EC670(__int64 a1)
{
  __int64 v1; // rbx
  unsigned int v2; // r15d
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned __int64 v9; // [rsp+8h] [rbp-78h]
  __int64 v10; // [rsp+18h] [rbp-68h] BYREF
  __int64 v11; // [rsp+20h] [rbp-60h] BYREF
  __int64 v12; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v13[2]; // [rsp+30h] [rbp-50h] BYREF
  char v14; // [rsp+40h] [rbp-40h]
  char v15; // [rsp+41h] [rbp-3Fh]

  v1 = sub_3909290(a1 + 144);
  if ( *(_BYTE *)(a1 + 845) || !(unsigned __int8)sub_38E36C0(a1) )
  {
    v13[0] = 0;
    if ( !sub_38EB6A0(a1, &v10, (__int64)v13) )
    {
      v4 = 0;
      v11 = 1;
      v12 = 0;
      v9 = 0;
      if ( !(unsigned __int8)sub_3909EB0(a1, 25)
        || (v5 = sub_3909460(a1), v4 = sub_39092A0(v5), !(unsigned __int8)sub_38EB9C0(a1, &v11))
        && (!(unsigned __int8)sub_3909EB0(a1, 25)
         || (v8 = sub_3909460(a1), v9 = sub_39092A0(v8), !(unsigned __int8)sub_38EB9C0(a1, &v12))) )
      {
        v15 = 1;
        v14 = 3;
        v13[0] = "unexpected token in '.fill' directive";
        v2 = sub_3909E20(a1, 9, v13);
        if ( !(_BYTE)v2 )
        {
          v6 = v11;
          if ( v11 < 0 )
          {
            v15 = 1;
            v13[0] = "'.fill' directive with negative size has no effect";
            v14 = 3;
            sub_38E4170((_QWORD *)a1, v4, (__int64)v13, 0, 0);
            return v2;
          }
          if ( v11 > 8 )
          {
            v15 = 1;
            v13[0] = "'.fill' directive with size greater than 8 has been truncated to 8";
            v14 = 3;
            sub_38E4170((_QWORD *)a1, v4, (__int64)v13, 0, 0);
            v7 = v12;
            v6 = 8;
            v11 = 8;
            if ( v12 == (unsigned int)v12 )
            {
LABEL_14:
              (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 328) + 504LL))(
                *(_QWORD *)(a1 + 328),
                v10,
                v6,
                v7,
                v1);
              return v2;
            }
          }
          else
          {
            v7 = v12;
            if ( v12 == (unsigned int)v12 || v11 <= 4 )
              goto LABEL_14;
          }
          v15 = 1;
          v13[0] = "'.fill' directive pattern has been truncated to 32-bits";
          v14 = 3;
          sub_38E4170((_QWORD *)a1, v9, (__int64)v13, 0, 0);
          v7 = v12;
          v6 = v11;
          goto LABEL_14;
        }
      }
    }
  }
  return 1;
}
