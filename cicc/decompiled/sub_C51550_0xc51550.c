// Function: sub_C51550
// Address: 0xc51550
//
void __fastcall sub_C51550(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // r13
  void (__fastcall *v7)(__int64, __int64, __int64); // rax
  void (__fastcall *v8)(__int64, __int64, __int64); // rax
  void (__fastcall *v9)(__int64, __int64, __int64); // rax
  __int64 v10; // rdi
  void (__fastcall *v11)(__int64, __int64, __int64); // rax
  __int64 v12; // rdi
  void (__fastcall *v13)(__int64, __int64, __int64); // rax
  __int64 v14; // rdi
  __int64 v15; // rdi
  void (__fastcall *v16)(__int64, __int64, __int64); // rax
  __int64 v17; // rdi
  void (__fastcall *v18)(__int64, __int64, __int64); // rax
  __int64 v19; // rdi
  void (__fastcall *v20)(__int64, __int64, __int64); // rax
  __int64 v21; // rdi

  if ( a1 )
  {
    *(_QWORD *)(a1 + 1504) = off_49DC520;
    v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1680);
    if ( v3 )
    {
      a2 = a1 + 1664;
      v3(a2, a2, 3);
    }
    if ( !*(_BYTE *)(a1 + 1628) )
      _libc_free(*(_QWORD *)(a1 + 1608), a2);
    v4 = *(_QWORD *)(a1 + 1576);
    if ( v4 != a1 + 1592 )
      _libc_free(v4, a2);
    v5 = *(_QWORD *)(a1 + 1480);
    v6 = *(_QWORD *)(a1 + 1472);
    if ( v5 != v6 )
    {
      do
      {
        v7 = *(void (__fastcall **)(__int64, __int64, __int64))(v6 + 16);
        if ( v7 )
        {
          a2 = v6;
          v7(v6, v6, 3);
        }
        v6 += 32;
      }
      while ( v5 != v6 );
      v6 = *(_QWORD *)(a1 + 1472);
    }
    if ( v6 )
    {
      a2 = *(_QWORD *)(a1 + 1488) - v6;
      j_j___libc_free_0(v6, a2);
    }
    v8 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1456);
    if ( v8 )
    {
      a2 = a1 + 1440;
      v8(a1 + 1440, a1 + 1440, 3);
    }
    *(_QWORD *)(a1 + 1240) = &unk_49DC090;
    v9 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1424);
    if ( v9 )
    {
      a2 = a1 + 1408;
      v9(a1 + 1408, a1 + 1408, 3);
    }
    if ( !*(_BYTE *)(a1 + 1364) )
      _libc_free(*(_QWORD *)(a1 + 1344), a2);
    v10 = *(_QWORD *)(a1 + 1312);
    if ( v10 != a1 + 1328 )
      _libc_free(v10, a2);
    *(_QWORD *)(a1 + 1040) = &unk_49DC090;
    v11 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1224);
    if ( v11 )
    {
      a2 = a1 + 1208;
      v11(a1 + 1208, a1 + 1208, 3);
    }
    if ( !*(_BYTE *)(a1 + 1164) )
      _libc_free(*(_QWORD *)(a1 + 1144), a2);
    v12 = *(_QWORD *)(a1 + 1112);
    if ( v12 != a1 + 1128 )
      _libc_free(v12, a2);
    v13 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1024);
    *(_QWORD *)(a1 + 848) = off_49DC4A0;
    if ( v13 )
    {
      a2 = a1 + 1008;
      v13(a1 + 1008, a1 + 1008, 3);
    }
    if ( !*(_BYTE *)(a1 + 972) )
      _libc_free(*(_QWORD *)(a1 + 952), a2);
    v14 = *(_QWORD *)(a1 + 920);
    if ( v14 != a1 + 936 )
      _libc_free(v14, a2);
    if ( !*(_BYTE *)(a1 + 828) )
      _libc_free(*(_QWORD *)(a1 + 808), a2);
    v15 = *(_QWORD *)(a1 + 776);
    if ( v15 != a1 + 792 )
      _libc_free(v15, a2);
    v16 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 688);
    *(_QWORD *)(a1 + 512) = off_49DC4A0;
    if ( v16 )
    {
      a2 = a1 + 672;
      v16(a1 + 672, a1 + 672, 3);
    }
    if ( !*(_BYTE *)(a1 + 636) )
      _libc_free(*(_QWORD *)(a1 + 616), a2);
    v17 = *(_QWORD *)(a1 + 584);
    if ( v17 != a1 + 600 )
      _libc_free(v17, a2);
    v18 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 496);
    *(_QWORD *)(a1 + 320) = off_49DC420;
    if ( v18 )
    {
      a2 = a1 + 480;
      v18(a1 + 480, a1 + 480, 3);
    }
    if ( !*(_BYTE *)(a1 + 444) )
      _libc_free(*(_QWORD *)(a1 + 424), a2);
    v19 = *(_QWORD *)(a1 + 392);
    if ( v19 != a1 + 408 )
      _libc_free(v19, a2);
    v20 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 304);
    *(_QWORD *)(a1 + 128) = off_49DC420;
    if ( v20 )
    {
      a2 = a1 + 288;
      v20(a1 + 288, a1 + 288, 3);
    }
    if ( !*(_BYTE *)(a1 + 252) )
      _libc_free(*(_QWORD *)(a1 + 232), a2);
    v21 = *(_QWORD *)(a1 + 200);
    if ( v21 != a1 + 216 )
      _libc_free(v21, a2);
    j_j___libc_free_0(a1, 1696);
  }
}
