// Function: sub_2BF9FF0
// Address: 0x2bf9ff0
//
__int64 __fastcall sub_2BF9FF0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r13
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi

  v1 = sub_2BF9BD0(a1);
  v2 = sub_2BF0C30(v1, *(_QWORD *)(a1 + 128));
  v3 = *(_QWORD *)(a1 + 120);
  v4 = v2;
  if ( a1 + 112 != v3 )
  {
    v5 = v2 + 112;
    do
    {
      if ( !v3 )
        BUG();
      v6 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v3 - 24) + 16LL))(v3 - 24);
      v6[10] = v4;
      v7 = v6[3];
      v8 = *(_QWORD *)(v4 + 112);
      v6[4] = v5;
      v8 &= 0xFFFFFFFFFFFFFFF8LL;
      v6[3] = v8 | v7 & 7;
      *(_QWORD *)(v8 + 8) = v6 + 3;
      *(_QWORD *)(v4 + 112) = *(_QWORD *)(v4 + 112) & 7LL | (unsigned __int64)(v6 + 3);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( a1 + 112 != v3 );
  }
  return v4;
}
