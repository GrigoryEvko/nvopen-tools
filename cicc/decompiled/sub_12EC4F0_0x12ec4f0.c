// Function: sub_12EC4F0
// Address: 0x12ec4f0
//
__int64 __fastcall sub_12EC4F0(__int64 a1, __int64 a2)
{
  int v2; // r14d
  _DWORD *v3; // rax
  __int64 v4; // rax
  _DWORD *v6; // rax
  __int64 v7; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v8[64]; // [rsp+10h] [rbp-40h] BYREF

  sub_1611EE0(v8);
  sub_12EA530(&v7, a1, a2);
  sub_12EA960(a1, (__int64)v8, a2, &v7);
  v2 = *(_DWORD *)(*(_QWORD *)(a1 + 1080) + 28LL);
  v3 = (_DWORD *)sub_1C42D70(4, 4);
  *v3 = v2;
  sub_16D40E0(qword_4FBB410, v3);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 1080) + 28LL) == 3 && !BYTE4(qword_4FBB370[2]) )
  {
    v6 = (_DWORD *)sub_1C42D70(4, 4);
    *v6 = 6;
    sub_16D40E0(qword_4FBB370, v6);
  }
  sub_12EC230(a1, (__int64)v8);
  v4 = sub_1654860(1);
  sub_1619140(v8, v4, 0);
  sub_160FB70(v8, ***(_QWORD ***)(a1 + 1080), *(_QWORD *)(**(_QWORD **)(a1 + 1080) + 8LL));
  sub_1619BD0(v8, a2);
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  return sub_160FE50(v8);
}
