// Function: sub_26EC760
// Address: 0x26ec760
//
__int64 __fastcall sub_26EC760(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 i; // [rsp+8h] [rbp-38h]

  if ( byte_4FF8A08 )
  {
    v4 = *(_QWORD *)(a3 + 32);
    for ( i = a3 + 24; i != v4; v4 = *(_QWORD *)(v4 + 8) )
    {
      v7 = 0;
      if ( v4 )
        v7 = v4 - 56;
      if ( !sub_B2FC80(v7) )
      {
        v8 = sub_BC0510(a4, &unk_4F82418, a3);
        sub_26EC370(a2, v7, *(_QWORD *)(v8 + 8));
      }
    }
  }
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
