// Function: sub_2337A80
// Address: 0x2337a80
//
__int64 __fastcall sub_2337A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 result; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  if ( *(_DWORD *)(a2 + 8) )
    sub_2303E40(a1, (char **)a2, a3, a4, a5, a6);
  *(_DWORD *)(a1 + 64) = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a1 + 72) = v6;
  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 80) = 0;
  *(_QWORD *)(a1 + 80) = v7;
  v8 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a2 + 88) = 0;
  *(_QWORD *)(a1 + 88) = v8;
  v9 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a2 + 96) = 0;
  *(_QWORD *)(a1 + 96) = v9;
  v10 = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a2 + 104) = 0;
  *(_QWORD *)(a1 + 104) = v10;
  result = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 112) = 0;
  *(_QWORD *)(a1 + 112) = result;
  return result;
}
