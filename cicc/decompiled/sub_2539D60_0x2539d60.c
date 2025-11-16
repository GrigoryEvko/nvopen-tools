// Function: sub_2539D60
// Address: 0x2539d60
//
__int64 __fastcall sub_2539D60(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v3; // rdi
  unsigned int v4; // eax
  unsigned int v5; // eax
  __int64 v7; // rdx
  __int64 v8; // rdx

  v1 = a1 + 48;
  v3 = a1 + 16;
  if ( *(_DWORD *)(a1 + 24) <= 0x40u && (v4 = *(_DWORD *)(a1 + 56), v4 <= 0x40) )
  {
    v8 = *(_QWORD *)(a1 + 48);
    *(_DWORD *)(a1 + 24) = v4;
    *(_QWORD *)(a1 + 16) = v8;
  }
  else
  {
    sub_C43990(v3, v1);
  }
  if ( *(_DWORD *)(a1 + 40) <= 0x40u && (v5 = *(_DWORD *)(a1 + 72), v5 <= 0x40) )
  {
    v7 = *(_QWORD *)(a1 + 64);
    *(_DWORD *)(a1 + 40) = v5;
    *(_QWORD *)(a1 + 32) = v7;
    return 0;
  }
  else
  {
    sub_C43990(a1 + 32, a1 + 64);
    return 0;
  }
}
