// Function: sub_1055B50
// Address: 0x1055b50
//
unsigned __int64 __fastcall sub_1055B50(unsigned __int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned __int64 v4; // r13
  __int64 *v7; // rax
  __int64 v9; // [rsp+8h] [rbp-A8h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v11; // [rsp+20h] [rbp-90h]
  __int64 v12; // [rsp+28h] [rbp-88h] BYREF
  unsigned int v13; // [rsp+30h] [rbp-80h]
  int v14; // [rsp+68h] [rbp-48h] BYREF
  __int64 (__fastcall *v15)(__int64 *, __int64); // [rsp+70h] [rbp-40h]
  __int64 *v16; // [rsp+78h] [rbp-38h]

  v4 = a1;
  if ( *(_DWORD *)(a2 + 20) != *(_DWORD *)(a2 + 24) )
  {
    v9 = a2;
    v10[1] = 0;
    v7 = &v12;
    v11 = 1;
    v10[0] = a3;
    do
    {
      *v7 = -4096;
      v7 += 2;
    }
    while ( v7 != (__int64 *)&v14 );
    v16 = &v9;
    v14 = 0;
    v15 = sub_1054B40;
    v4 = sub_1054C20((__int64)v10, a1);
    if ( (v11 & 1) == 0 )
      sub_C7D6A0(v12, 16LL * v13, 8);
    if ( a1 != sub_1055AA0(v4, a2, a3) && a4 )
      return 0;
  }
  return v4;
}
