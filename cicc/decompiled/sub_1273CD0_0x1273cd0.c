// Function: sub_1273CD0
// Address: 0x1273cd0
//
__int64 __fastcall sub_1273CD0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  bool v5; // zf
  __int64 result; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8; // [rsp+10h] [rbp-30h]

  if ( (unsigned __int8)sub_15E4F60(a2) )
    sub_127B550("globals that are not defined cannot force usage!");
  v7[0] = 4;
  v7[1] = 0;
  v8 = a2;
  if ( a2 != -8 && a2 != 0 && a2 != -16 )
    sub_164C220(v7);
  v3 = *(_QWORD *)(a1 + 432);
  if ( v3 == *(_QWORD *)(a1 + 440) )
  {
    sub_1273A40((char **)(a1 + 424), (char *)v3, v7);
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = 4;
      *(_QWORD *)(v3 + 8) = 0;
      v4 = v8;
      v5 = v8 == -8;
      *(_QWORD *)(v3 + 16) = v8;
      if ( v4 != 0 && !v5 && v4 != -16 )
        sub_1649AC0(v3, v7[0] & 0xFFFFFFFFFFFFFFF8LL);
      v3 = *(_QWORD *)(a1 + 432);
    }
    *(_QWORD *)(a1 + 432) = v3 + 24;
  }
  result = v8;
  if ( v8 != 0 && v8 != -8 && v8 != -16 )
    return sub_1649B30(v7);
  return result;
}
