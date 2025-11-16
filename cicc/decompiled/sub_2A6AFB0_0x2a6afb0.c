// Function: sub_2A6AFB0
// Address: 0x2a6afb0
//
void __fastcall sub_2A6AFB0(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  unsigned __int8 *v4; // rax
  __int64 v5; // rax
  unsigned int v6; // r15d
  unsigned __int8 *v7; // rsi
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // rax
  __int64 *v11; // rax
  bool v12; // al
  __int64 *v13; // rax
  bool v14; // al
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  bool v19; // [rsp+Fh] [rbp-E1h]
  __int64 v20; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int8 v21[48]; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int8 v22[48]; // [rsp+60h] [rbp-90h] BYREF
  __int64 v23[12]; // [rsp+90h] [rbp-60h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 15
    || (v3 = a1 + 136, v23[0] = a2, *(_BYTE *)sub_2A686D0(a1 + 136, v23) == 6) )
  {
    sub_2A6A450(a1, a2);
  }
  else
  {
    v4 = (unsigned __int8 *)sub_2A68BC0(a1, *(unsigned __int8 **)(a2 - 96));
    sub_22C05A0((__int64)v21, v4);
    if ( v21[0] <= 1u )
    {
LABEL_10:
      sub_22C0090(v21);
      return;
    }
    v5 = sub_2A637C0(a1, (__int64)v21, *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL));
    if ( v5 && *(_BYTE *)v5 == 17 )
    {
      v6 = *(_DWORD *)(v5 + 32);
      if ( v6 <= 0x40 )
      {
        if ( !*(_QWORD *)(v5 + 24) )
          goto LABEL_8;
      }
      else if ( v6 == (unsigned int)sub_C444A0(v5 + 24) )
      {
LABEL_8:
        v7 = *(unsigned __int8 **)(a2 - 32);
LABEL_9:
        v8 = (unsigned __int8 *)sub_2A68BC0(a1, v7);
        sub_22C05A0((__int64)v23, v8);
        sub_2A689D0(a1, a2, (unsigned __int8 *)v23, 0x100000000LL);
        sub_22C0090((unsigned __int8 *)v23);
        goto LABEL_10;
      }
      v7 = *(unsigned __int8 **)(a2 - 64);
      goto LABEL_9;
    }
    v9 = (unsigned __int8 *)sub_2A68BC0(a1, *(unsigned __int8 **)(a2 - 64));
    sub_22C05A0((__int64)v22, v9);
    v10 = (unsigned __int8 *)sub_2A68BC0(a1, *(unsigned __int8 **)(a2 - 32));
    sub_22C05A0((__int64)v23, v10);
    v20 = a2;
    v11 = sub_2A686D0(v3, &v20);
    v12 = sub_2A625F0((__int64)v11, (__int64)v22, 0, 0, 1u);
    v20 = a2;
    v19 = v12;
    v13 = sub_2A686D0(v3, &v20);
    v14 = sub_2A625F0((__int64)v13, (__int64)v23, 0, 0, 1u);
    if ( v19 || v14 )
    {
      v20 = a2;
      v15 = sub_2A686D0(v3, &v20);
      sub_2A63310(a1, v15, a2, v16, v17, v18);
    }
    sub_22C0090((unsigned __int8 *)v23);
    sub_22C0090(v22);
    sub_22C0090(v21);
  }
}
