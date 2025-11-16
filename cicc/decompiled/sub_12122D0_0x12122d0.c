// Function: sub_12122D0
// Address: 0x12122d0
//
__int64 __fastcall sub_12122D0(__int64 a1, _QWORD *a2, _DWORD *a3)
{
  int v5; // eax
  char v6; // r14
  char v7; // r15
  __int64 result; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rdx

  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 == 216 )
  {
    v6 = 1;
    v7 = 0;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  else
  {
    v6 = 0;
    v7 = 0;
    if ( v5 == 243 )
    {
      v7 = 1;
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    }
  }
  result = sub_120AFE0(a1, 506, "expected GV ID");
  if ( !(_BYTE)result )
  {
    v9 = *(unsigned int *)(a1 + 280);
    *a3 = v9;
    v10 = *(_QWORD *)(a1 + 1624);
    if ( v9 < (*(_QWORD *)(a1 + 1632) - v10) >> 3
      && (v11 = *(_QWORD *)(v10 + 8 * v9), (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0) )
    {
      *a2 = v11;
    }
    else
    {
      *a2 = -8;
    }
    if ( v6 )
      *a2 |= 2uLL;
    if ( v7 )
      *a2 |= 4uLL;
    else
      return 0;
  }
  return result;
}
