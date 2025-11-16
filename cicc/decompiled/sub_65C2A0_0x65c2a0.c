// Function: sub_65C2A0
// Address: 0x65c2a0
//
__int64 __fastcall sub_65C2A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 v6; // r13
  _QWORD v7[2]; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v8; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v9; // [rsp+18h] [rbp-98h]
  __int64 v10; // [rsp+20h] [rbp-90h]
  _BYTE v11[128]; // [rsp+30h] [rbp-80h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 32LL);
  if ( v2 && !(unsigned int)sub_8DBE70(a2) )
  {
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v5 = sub_823970(0);
    v9 = 0;
    v8 = v5;
    v7[0] = 0;
    v7[1] = 0;
    sub_892150(v11);
    v3 = sub_6F1D40(a2, v2, &v8, v11, v7);
    if ( v3 )
    {
      v3 = 1;
    }
    else
    {
      v6 = sub_67DA80(3125, v2 + 28, a2);
      sub_67E370(v6, v7);
      sub_685910(v6);
    }
    sub_823A00(v8, 24 * v9);
  }
  else
  {
    return 1;
  }
  return v3;
}
