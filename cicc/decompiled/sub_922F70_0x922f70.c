// Function: sub_922F70
// Address: 0x922f70
//
__int64 __fastcall sub_922F70(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r14
  __int64 v5; // rsi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx

  v3 = (__int64)a3 + 36;
  v5 = a3[7];
  if ( *(_BYTE *)(v5 + 173) != 2 )
    sub_91B8A0("cannot generate l-value for this constant!", (_DWORD *)a3 + 9, 1);
  v7 = sub_90A830(*(__int64 **)(a2 + 32), v5, 0);
  v8 = sub_92CAE0(a2, v7, v3);
  v9 = *a3;
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = v9;
  *(_DWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  return a1;
}
