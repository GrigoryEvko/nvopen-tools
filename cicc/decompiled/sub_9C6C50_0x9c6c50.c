// Function: sub_9C6C50
// Address: 0x9c6c50
//
__int64 __fastcall sub_9C6C50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int16 a6)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  v8 = sub_BD2C40(104, unk_3F10A14);
  if ( v8 )
  {
    v9 = sub_B501B0(*(_QWORD *)(a1 + 8), a2, a3);
    sub_B44260(v8, v9, 64, 1, a5, a6);
    if ( *(_QWORD *)(v8 - 32) )
    {
      v10 = *(_QWORD *)(v8 - 24);
      **(_QWORD **)(v8 - 16) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v8 - 16);
    }
    *(_QWORD *)(v8 - 32) = a1;
    v11 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(v8 - 24) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = v8 - 24;
    *(_QWORD *)(v8 - 16) = a1 + 16;
    *(_QWORD *)(a1 + 16) = v8 - 32;
    *(_QWORD *)(v8 + 72) = v8 + 88;
    *(_QWORD *)(v8 + 80) = 0x400000000LL;
    sub_B50030(v8, a2, a3, a4);
  }
  return v8;
}
