// Function: sub_EA87E0
// Address: 0xea87e0
//
__int64 __fastcall sub_EA87E0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v10; // rax

  if ( a2[14] == 33 && a2[17] == 15 )
  {
    v10 = sub_22077B0(936);
    v8 = v10;
    if ( v10 )
    {
      sub_EA72E0(v10, a1, a2, a3, a4, a5);
      *(_QWORD *)(v8 + 928) = a3;
      *(_QWORD *)v8 = off_49E48D0;
      *(_QWORD *)(v8 + 920) = v8 + 40;
      *(_BYTE *)(v8 + 152) = 0;
      *(_BYTE *)(v8 + 154) = 1;
      *(_WORD *)(v8 + 168) = 257;
    }
  }
  else
  {
    v7 = sub_22077B0(920);
    v8 = v7;
    if ( v7 )
      sub_EA72E0(v7, a1, a2, a3, a4, a5);
  }
  return v8;
}
