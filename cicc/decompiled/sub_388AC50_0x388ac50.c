// Function: sub_388AC50
// Address: 0x388ac50
//
__int64 __fastcall sub_388AC50(_QWORD *a1)
{
  __int64 v1; // rbp
  __int64 result; // rax
  __int64 v3; // rax
  int v4; // edx
  int v5; // edx
  int v6; // edx
  __int64 v7; // [rsp-68h] [rbp-68h]
  _QWORD v8[2]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v9; // [rsp-38h] [rbp-38h]
  _QWORD v10[2]; // [rsp-28h] [rbp-28h] BYREF
  __int16 v11; // [rsp-18h] [rbp-18h]
  __int64 v12; // [rsp-8h] [rbp-8h]

  if ( !a1[23] )
    return 0;
  v12 = v1;
  if ( a1[158] )
  {
    v3 = a1[156];
    v4 = *(_DWORD *)(v3 + 32);
    v8[0] = "use of undefined summary '^";
    LODWORD(v7) = v4;
LABEL_7:
    v10[1] = "'";
    v8[1] = v7;
    v9 = 2307;
    v10[0] = v8;
    v11 = 770;
    return sub_38814C0((__int64)(a1 + 1), *(_QWORD *)(*(_QWORD *)(v3 + 40) + 8LL), (__int64)v10);
  }
  if ( a1[164] )
  {
    v3 = a1[162];
    v5 = *(_DWORD *)(v3 + 32);
    v8[0] = "use of undefined summary '^";
    LODWORD(v7) = v5;
    goto LABEL_7;
  }
  result = 0;
  if ( a1[173] )
  {
    v3 = a1[171];
    v6 = *(_DWORD *)(v3 + 32);
    v8[0] = "use of undefined type id summary '^";
    LODWORD(v7) = v6;
    goto LABEL_7;
  }
  return result;
}
