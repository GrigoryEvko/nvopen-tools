// Function: sub_1F3C8D0
// Address: 0x1f3c8d0
//
__int64 __fastcall sub_1F3C8D0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v13; // [rsp-10h] [rbp-50h]

  v10 = *(_QWORD *)(sub_1E15F70(a2) + 40);
  if ( a3 == 2 )
    goto LABEL_9;
  if ( a3 > 2 )
  {
    if ( a3 == 3 )
      goto LABEL_5;
LABEL_8:
    v11 = 0;
    goto LABEL_6;
  }
  if ( !a3 )
  {
LABEL_9:
    v11 = sub_1E69D60(v10, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
    goto LABEL_6;
  }
  if ( a3 != 1 )
    goto LABEL_8;
LABEL_5:
  v11 = sub_1E69D60(v10, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
LABEL_6:
  sub_1F3BF50(a1, a2, v11, a3, a4, a5, a6);
  return v13;
}
