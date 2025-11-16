// Function: sub_1614540
// Address: 0x1614540
//
__int64 __fastcall sub_1614540(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // r14
  _QWORD *v6; // rdi
  _QWORD *v7; // rsi
  unsigned __int8 v9; // [rsp+Fh] [rbp-41h]
  __int64 v10[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = sub_16135E0(*(_QWORD *)(a1 + 16), a2);
  v9 = *(_BYTE *)(v2 + 160);
  if ( v9 )
    return 1;
  v3 = *(_QWORD **)(a1 + 256);
  v4 = &v3[*(unsigned int *)(a1 + 264)];
  if ( v4 == v3 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v5 = *v3;
      if ( !(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v3 + 112LL))(*v3) )
      {
        v6 = *(_QWORD **)(v2 + 112);
        v10[0] = *(_QWORD *)(v5 + 16);
        v7 = &v6[*(unsigned int *)(v2 + 120)];
        if ( v7 == sub_160D180(v6, (__int64)v7, v10) )
          break;
      }
      if ( v4 == ++v3 )
        return 1;
    }
  }
  return v9;
}
