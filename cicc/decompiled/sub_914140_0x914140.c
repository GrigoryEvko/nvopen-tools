// Function: sub_914140
// Address: 0x914140
//
__int64 __fastcall sub_914140(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  bool v5; // zf
  __int64 result; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8; // [rsp+10h] [rbp-30h]

  if ( (unsigned __int8)sub_B2FC80(a2) )
    sub_91B8A0("globals that are not defined cannot force usage!");
  v7[0] = 4;
  v7[1] = 0;
  v8 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
    sub_BD73F0(v7);
  v3 = *(_QWORD *)(a1 + 416);
  if ( v3 == *(_QWORD *)(a1 + 424) )
  {
    sub_913E90((char **)(a1 + 408), (char *)v3, v7);
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = 4;
      *(_QWORD *)(v3 + 8) = 0;
      v4 = v8;
      v5 = v8 == 0;
      *(_QWORD *)(v3 + 16) = v8;
      if ( v4 != -4096 && !v5 && v4 != -8192 )
        sub_BD6050(v3, v7[0] & 0xFFFFFFFFFFFFFFF8LL);
      v3 = *(_QWORD *)(a1 + 416);
    }
    *(_QWORD *)(a1 + 416) = v3 + 24;
  }
  result = v8;
  if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
    return sub_BD60C0(v7);
  return result;
}
