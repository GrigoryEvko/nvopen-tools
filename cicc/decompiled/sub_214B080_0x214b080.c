// Function: sub_214B080
// Address: 0x214b080
//
__int64 __fastcall sub_214B080(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // r12
  bool v5; // zf
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a2;
  *a2 = 0;
  v3 = sub_22077B0(920);
  v4 = v3;
  if ( v3 )
  {
    v7[0] = v2;
    sub_396D9B0(v3, a1, v7);
    if ( v7[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v7[0] + 48LL))(v7[0]);
    *(_QWORD *)(v4 + 760) = 0;
    v5 = *(_DWORD *)(a1 + 952) == 1;
    *(_QWORD *)v4 = off_4A01708;
    *(_QWORD *)(v4 + 752) = v4 + 768;
    *(_BYTE *)(v4 + 768) = 0;
    *(_QWORD *)(v4 + 808) = 0;
    *(_QWORD *)(v4 + 816) = 0;
    *(_QWORD *)(v4 + 824) = 0;
    *(_DWORD *)(v4 + 832) = 0;
    *(_QWORD *)(v4 + 840) = 0;
    *(_DWORD *)(v4 + 856) = 0;
    *(_QWORD *)(v4 + 864) = 0;
    *(_QWORD *)(v4 + 872) = v4 + 856;
    *(_QWORD *)(v4 + 880) = v4 + 856;
    *(_QWORD *)(v4 + 888) = 0;
    *(_QWORD *)(v4 + 904) = 0;
    *(_BYTE *)(v4 + 896) = v5;
    return v4;
  }
  if ( !v2 )
    return v4;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
  return 0;
}
