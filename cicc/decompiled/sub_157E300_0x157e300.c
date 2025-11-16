// Function: sub_157E300
// Address: 0x157e300
//
__int64 __fastcall sub_157E300(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  result = sub_1568D80(a1, (void **)&v6);
  if ( (_BYTE)result )
  {
    v3 = *(_QWORD *)(a1 + 8);
    while ( v3 )
    {
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      v5 = sub_1648700(v4);
      if ( *(_BYTE *)(v5 + 16) == 78 )
        sub_156E800(v5, v6);
    }
    return sub_15E3D00(a1);
  }
  return result;
}
