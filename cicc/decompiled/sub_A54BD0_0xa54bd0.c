// Function: sub_A54BD0
// Address: 0xa54bd0
//
__int64 __fastcall sub_A54BD0(__int64 a1, __int64 a2)
{
  int v2; // eax
  bool v3; // zf
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // rsi

  *(_QWORD *)a1 = &unk_49DC840;
  *(_DWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 44) = 1;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 96;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 4;
  v2 = *(_DWORD *)(a2 + 44);
  *(_BYTE *)(a1 + 104) = 0;
  v3 = v2 == 0;
  *(_QWORD *)(a1 + 48) = a2;
  v4 = *(_QWORD *)(a2 + 16);
  if ( v3 || v4 )
  {
    v5 = *(_QWORD *)(a2 + 24) - v4;
    if ( !v5 )
    {
LABEL_4:
      sub_CB5980(a1, 0, 0, 0);
      goto LABEL_5;
    }
  }
  else
  {
    v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 88LL))(a2);
    if ( !v5 )
    {
      if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(a1 + 16) )
        sub_CB5AE0(a1);
      goto LABEL_4;
    }
    if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(a1 + 16) )
      sub_CB5AE0(a1);
  }
  v8 = sub_2207820(v5);
  sub_CB5980(a1, v8, v5, 1);
LABEL_5:
  v6 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(v6 + 32) != *(_QWORD *)(v6 + 16) )
    sub_CB5AE0(*(_QWORD *)(a1 + 48));
  sub_CB5980(v6, 0, 0, 0);
  result = *(unsigned __int8 *)(*(_QWORD *)(a1 + 48) + 40LL);
  *(_QWORD *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 40) = result;
  return result;
}
