// Function: sub_71B300
// Address: 0x71b300
//
__int64 __fastcall sub_71B300(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi

  v3 = a1;
  if ( a2 )
  {
    v4 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_QWORD *)(a3 + 120) )
      *(_QWORD *)(*(_QWORD *)(v4 + 288) + 112LL) = a2;
    else
      *(_QWORD *)(a3 + 120) = a2;
    *(_QWORD *)(v4 + 288) = a2;
    sub_72EE40(a2, 7, a3);
    v5 = sub_726B30(20);
    v6 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = v5;
    *(_QWORD *)(v5 + 24) = v6;
    v7 = *(_QWORD **)(a1 + 16);
    v8 = sub_727640();
    v7[9] = v8;
    *(_BYTE *)(v8 + 8) = 7;
    *(_QWORD *)(v7[9] + 16LL) = a2;
    v9 = sub_726B30(17);
    v10 = v7[3];
    v7[2] = v9;
    *(_QWORD *)(v9 + 24) = v10;
    v3 = v7[2];
    v11 = *(_QWORD *)(a2 + 184);
    *(_QWORD *)(v3 + 72) = v11;
    sub_7340D0(v11, 0, 1);
  }
  return v3;
}
