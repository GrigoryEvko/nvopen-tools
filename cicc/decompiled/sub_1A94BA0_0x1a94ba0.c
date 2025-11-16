// Function: sub_1A94BA0
// Address: 0x1a94ba0
//
__int64 __fastcall sub_1A94BA0(__int64 a1)
{
  __int64 *v1; // rax
  unsigned int v2; // r8d
  __int64 v3; // rdx
  __int64 v4; // rax
  int v6; // eax

  v1 = (__int64 *)sub_15E0FA0(a1);
  v3 = *v1;
  v4 = v1[1];
  if ( v4 == 18 )
  {
    if ( !(*(_QWORD *)v3 ^ 0x696F706574617473LL | *(_QWORD *)(v3 + 8) ^ 0x706D6178652D746ELL)
      && *(_WORD *)(v3 + 16) == 25964 )
    {
      v6 = 0;
      goto LABEL_6;
    }
  }
  else
  {
    v2 = 0;
    if ( v4 != 7 )
      return 0;
    if ( *(_DWORD *)v3 == 1701998435 && *(_WORD *)(v3 + 4) == 27747 && *(_BYTE *)(v3 + 6) == 114 )
    {
      v6 = 0;
      goto LABEL_6;
    }
  }
  v6 = 1;
LABEL_6:
  LOBYTE(v2) = v6 == 0;
  return v2;
}
