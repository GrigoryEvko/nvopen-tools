// Function: sub_120A540
// Address: 0x120a540
//
__int64 __fastcall sub_120A540(_QWORD *a1)
{
  __int64 v1; // rbp
  __int64 result; // rax
  __int64 v3; // rax
  int v4; // edx
  const char *v5; // rcx
  int v6; // edx
  _QWORD v7[2]; // [rsp-68h] [rbp-68h] BYREF
  int v8; // [rsp-58h] [rbp-58h]
  __int16 v9; // [rsp-48h] [rbp-48h]
  _QWORD *v10; // [rsp-38h] [rbp-38h] BYREF
  char *v11; // [rsp-28h] [rbp-28h]
  __int16 v12; // [rsp-18h] [rbp-18h]
  __int64 v13; // [rsp-8h] [rbp-8h]

  if ( !a1[44] )
    return 0;
  v13 = v1;
  if ( a1[196] )
  {
    v3 = a1[194];
    v4 = *(_DWORD *)(v3 + 32);
    v7[0] = "use of undefined summary '^";
    v9 = 2307;
    v8 = v4;
    v10 = v7;
    v11 = "'";
LABEL_7:
    v12 = 770;
    sub_11FD800((__int64)(a1 + 22), *(_QWORD *)(*(_QWORD *)(v3 + 40) + 8LL), (__int64)&v10, 1);
    return 1;
  }
  if ( a1[202] )
  {
    v3 = a1[200];
    v5 = "use of undefined summary '^";
    v6 = *(_DWORD *)(v3 + 32);
LABEL_9:
    v7[0] = v5;
    v8 = v6;
    v9 = 2307;
    v10 = v7;
    v11 = "'";
    goto LABEL_7;
  }
  result = 0;
  if ( a1[211] )
  {
    v3 = a1[209];
    v5 = "use of undefined type id summary '^";
    v6 = *(_DWORD *)(v3 + 32);
    goto LABEL_9;
  }
  return result;
}
